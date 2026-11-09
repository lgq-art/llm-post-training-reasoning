"""
Stage 3: 群组相对策略优化 (GRPO, Group Relative Policy Optimization) 训练器.

科研教学要点（对应 RQ3：DeepSeek-R1 范式核心实现）：
传统的 PPO 算法需要训练一个与策略模型同尺寸的 Value/Critic 网络来估计基线价值 $V(s)$，显存开销极大。
**GRPO (Shao et al., DeepSeekMath / DeepSeek-R1)** 通过“群组采样平均”彻底消除了 Critic 网络：
1. **群组采样 (Group Rollout)**：对每个数学问题 $q$，让当前策略 $\\pi_\\theta$ 采样生成 $G$ 条不同的解答轨迹 $\\{o_1, o_2, \\dots, o_G\\}$。
2. **可验证规则奖励 (Verifiable Rule-based Rewards)**：
   不依赖容易被钻空子（Reward Hacking）的神经网络奖励模型，而是直接用确定性规则打分：
   - **答案准确性奖励 $R_{\\text{acc}} \\in \\{0, 1\\}$**：提取 `\\boxed{}` 内的数值与标准答案比对。
   - **结构化格式奖励 $R_{\\text{fmt}} \\in [0, 0.3]$**：检查是否正确闭合 `<think>...</think><answer>...</answer>` 及分步推导结构。
3. **组内相对优势归一化 (Group Relative Advantage)**：
   $$A_i = \\frac{r_i - \\text{mean}(r_1, \\dots, r_G)}{\\text{std}(r_1, \\dots, r_G) + \\epsilon}$$
   比组内平均分高的轨迹获得正梯度激励（$A_i > 0$），比平均分低的轨迹受到抑制（$A_i < 0$）。
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple
import torch
from torch.optim import AdamW
from ..data.prompt_templates import (
    extract_predicted_answer,
    is_answer_correct,
    check_format_compliance,
)


def compute_rule_rewards(
    completions: List[str],
    gold_answer: str,
    acc_weight: float = 1.0,
    fmt_weight: float = 0.2,
    step_weight: float = 0.1,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """
    针对同一问题的 G 条生成轨迹，计算可验证规则奖励 (Verifiable Rule-based Rewards).
    """
    rewards = []
    details = []

    for text in completions:
        pred = extract_predicted_answer(text)
        correct = is_answer_correct(pred, gold_answer)
        r_acc = 1.0 if correct else 0.0

        fmt_info = check_format_compliance(text)
        # 渐进式格式奖励：有 <think> +0.3，有 <answer> +0.3，有 \boxed{} +0.4
        r_fmt = (
            (0.3 if fmt_info["has_think"] else 0.0)
            + (0.3 if fmt_info["has_answer"] else 0.0)
            + (0.4 if fmt_info["has_boxed"] else 0.0)
        )
        # 推理深度奖励：鼓励适度分步思考（15 ~ 180 词之间得满分，防止空洞或无限重复）
        words = fmt_info["reasoning_word_count"]
        if 15 <= words <= 180:
            r_step = 1.0
        elif words > 5:
            r_step = 0.5
        else:
            r_step = 0.0

        total_r = acc_weight * r_acc + fmt_weight * r_fmt + step_weight * r_step
        rewards.append(total_r)
        details.append(
            {
                "total_reward": round(total_r, 4),
                "acc_reward": r_acc,
                "fmt_reward": round(r_fmt, 4),
                "step_reward": r_step,
                "reasoning_words": words,
            }
        )

    return torch.tensor(rewards, dtype=torch.float32), details


def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    计算 GRPO 组内归一化优势值 $A_i = (r_i - \\mu_G) / (\\sigma_G + \\epsilon)$.
    """
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False)
    if std_r.item() < eps:
        return torch.zeros_like(rewards)
    return (rewards - mean_r) / (std_r + eps)


def run_grpo_training(
    model: torch.nn.Module,
    tokenizer: Any,
    grpo_dataset: Any,
    output_dir: str,
    group_size: int = 4,
    temperature: float = 0.8,
    kl_coef: float = 0.04,
    learning_rate: float = 5e-6,
    max_prompt_length: int = 256,
    max_completion_length: int = 128,
    max_steps: int = 20,
) -> Dict[str, Any]:
    """
    执行 Stage 3 GRPO 在线采样与组内相对策略优化训练.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    use_peft_disable = hasattr(model, "disable_adapter")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    history: List[Dict[str, float]] = []
    start_time = time.time()
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    total_samples = len(grpo_dataset)
    steps_to_run = min(max_steps, total_samples) if max_steps > 0 else total_samples

    for step_idx in range(steps_to_run):
        item = grpo_dataset[step_idx % total_samples]
        prompt_text = item["prompt"]
        gold_answer = item["gold_answer"]

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)[:max_prompt_length]
        prompt_tensor = torch.tensor(
            [prompt_ids] * group_size, dtype=torch.long, device=device
        )
        prompt_mask = torch.ones_like(prompt_tensor)
        prompt_len = prompt_tensor.shape[1]

        # 1. 群组采样 (Group Rollout): 采样 G 条不同的解答轨迹
        model.eval()
        with torch.no_grad():
            gen_sequences = model.generate(
                input_ids=prompt_tensor,
                attention_mask=prompt_mask,
                max_new_tokens=max_completion_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=pad_id,
            )

        completion_ids = gen_sequences[:, prompt_len:]
        completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # 2. 规则奖励打分与组内相对优势 A_i 计算
        rewards, reward_details = compute_rule_rewards(completions_text, gold_answer)
        rewards = rewards.to(device)
        advantages = compute_group_advantages(rewards).to(device)

        # 3. 策略梯度与 KL 散度正则化更新
        model.train()
        attention_mask = (gen_sequences != pad_id).long()
        outputs = model(input_ids=gen_sequences, attention_mask=attention_mask)

        # 提取生成部分的每 token 对数概率 (shape: [G, L_comp])
        comp_len = completion_ids.shape[1]
        if comp_len == 0:
            continue

        shift_logits = outputs.logits[:, prompt_len - 1 : -1, :]
        log_probs = torch.gather(
            shift_logits.log_softmax(dim=-1),
            dim=-1,
            index=completion_ids.unsqueeze(-1),
        ).squeeze(-1)

        # 计算参考策略对数概率以估计 Schulman 无偏 KL 近似: exp(log_ref - log_pol) - (log_ref - log_pol) - 1
        with torch.no_grad():
            if use_peft_disable:
                with model.disable_adapter():
                    ref_out = model(input_ids=gen_sequences, attention_mask=attention_mask)
            else:
                ref_out = outputs
            ref_shift_logits = ref_out.logits[:, prompt_len - 1 : -1, :]
            ref_log_probs = torch.gather(
                ref_shift_logits.log_softmax(dim=-1),
                dim=-1,
                index=completion_ids.unsqueeze(-1),
            ).squeeze(-1)

        log_ratio = ref_log_probs - log_probs
        per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0

        # GRPO 目标：最大化 A_i * log_probs - kl_coef * KL
        comp_mask = (completion_ids != pad_id).float()
        seq_log_prob = (log_probs * comp_mask).sum(dim=-1) / comp_mask.sum(dim=-1).clamp(min=1.0)
        seq_kl = (per_token_kl * comp_mask).sum(dim=-1) / comp_mask.sum(dim=-1).clamp(min=1.0)

        pg_loss = -(advantages.detach() * seq_log_prob).mean()
        kl_loss = kl_coef * seq_kl.mean()
        total_loss = pg_loss + kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_acc = sum(d["acc_reward"] for d in reward_details) / len(reward_details)
        avg_fmt = sum(d["fmt_reward"] for d in reward_details) / len(reward_details)
        avg_words = sum(d["reasoning_words"] for d in reward_details) / len(reward_details)

        history.append(
            {
                "step": step_idx + 1,
                "loss": round(total_loss.item(), 4),
                "mean_reward": round(rewards.mean().item(), 4),
                "accuracy_rate": round(avg_acc, 4),
                "format_score": round(avg_fmt, 4),
                "avg_reasoning_words": round(avg_words, 2),
                "kl_div": round(seq_kl.mean().item(), 5),
            }
        )

    elapsed = round(time.time() - start_time, 2)
    adapter_save_path = os.path.join(output_dir, "final_grpo_adapter")
    os.makedirs(adapter_save_path, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(adapter_save_path)

    summary = {
        "stage": "grpo_rl",
        "group_size": group_size,
        "kl_coef": kl_coef,
        "total_steps": len(history),
        "final_metrics": history[-1] if history else {},
        "elapsed_sec": elapsed,
        "history": history,
        "adapter_path": adapter_save_path,
    }
    with open(os.path.join(output_dir, "grpo_training_log.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
