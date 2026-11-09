"""
Stage 2: 离线偏好对齐 (DPO vs. SimPO) 训练器.

科研教学要点（对应 RQ2 核心公式实现）：
为什么要在同一个训练器中对比 DPO 和 SimPO？
1. **DPO (Direct Preference Optimization, Rafailov et al., NeurIPS 2023)**:
   需要同时在显存中维护当前策略模型 $\\pi_\\theta$ 和冻结的参考模型 $\\pi_{\\text{ref}}$：
   $$\\mathcal{L}_{\\text{DPO}} = -\\log \\sigma \\left( \\beta \\left[ (\\log \\pi_\\theta(y_w|x) - \\log \\pi_{\\text{ref}}(y_w|x)) - (\\log \\pi_\\theta(y_l|x) - \\log \\pi_{\\text{ref}}(y_l|x)) \\right] \\right)$$
   缺点：一是显存翻倍（或需要反复切换 LoRA 开关计算两次前向传播）；二是未对回答长度 $|y|$ 归一化，容易产生“越长分越高”的长度偏见（Length Bias）。

2. **SimPO (Simple Preference Optimization, Meng et al., NeurIPS 2024)**:
   直接用**长度归一化后的平均对数似然**作为隐式奖励，无需参考模型 $\\pi_{\\text{ref}}$，并引入目标奖励边际 $\\gamma > 0$：
   $$\\mathcal{L}_{\\text{SimPO}} = -\\log \\sigma \\left( \\frac{\\beta}{|y_w|} \\log \\pi_\\theta(y_w|x) - \\frac{\\beta}{|y_l|} \\log \\pi_\\theta(y_l|x) - \\gamma \\right)$$
   优点：省去参考模型前向计算（训练提速约 25%~35%），且天然抑制冗余废话！
"""

import os
import json
import time
import copy
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW


def get_sequence_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算每个样本在 Completion 部分的：
    1. 累积对数概率 $\\sum_{t} \\log P(y_t | x, y_{<t})$ (用于 DPO)
    2. 长度归一化平均对数概率 $\\frac{1}{|y|} \\sum_{t} \\log P(y_t | x, y_{<t})$ (用于 SimPO)
    """
    # Causal LM 错位对齐：用前 T-1 个位置的 logits 预测后 T-1 个位置的 token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_mask = shift_labels != -100
    safe_labels = shift_labels.clone()
    safe_labels[~loss_mask] = 0

    per_token_logps = torch.gather(
        shift_logits.log_softmax(-1),
        dim=2,
        index=safe_labels.unsqueeze(2),
    ).squeeze(2)

    masked_logps = per_token_logps * loss_mask.float()
    sum_logps = masked_logps.sum(dim=-1)
    valid_lengths = loss_mask.sum(dim=-1).clamp(min=1).float()
    avg_logps = sum_logps / valid_lengths

    return sum_logps, avg_logps


def compute_dpo_or_simpo_loss(
    policy_chosen_sum_logps: torch.Tensor,
    policy_chosen_avg_logps: torch.Tensor,
    policy_rejected_sum_logps: torch.Tensor,
    policy_rejected_avg_logps: torch.Tensor,
    ref_chosen_sum_logps: Optional[torch.Tensor] = None,
    ref_rejected_sum_logps: Optional[torch.Tensor] = None,
    algorithm: str = "simpo",
    beta: float = 2.0,
    simpo_gamma: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    核心数学函数：统一计算 DPO 或 SimPO 损失，并返回隐式奖励准确率 (Reward Accuracy) 与奖励差值 (Reward Margin).
    """
    algo = algorithm.lower()
    if algo == "simpo":
        # SimPO: 使用平均对数概率 avg_logps，无需参考模型
        chosen_rewards = beta * policy_chosen_avg_logps
        rejected_rewards = beta * policy_rejected_avg_logps
        logits = (chosen_rewards - rejected_rewards) - simpo_gamma
        losses = -F.logsigmoid(logits)
    elif algo == "dpo":
        if ref_chosen_sum_logps is None or ref_rejected_sum_logps is None:
            raise ValueError("DPO requires reference model log probabilities.")
        chosen_rewards = beta * (policy_chosen_sum_logps - ref_chosen_sum_logps)
        rejected_rewards = beta * (policy_rejected_sum_logps - ref_rejected_sum_logps)
        logits = chosen_rewards - rejected_rewards
        losses = -F.logsigmoid(logits)
    else:
        raise ValueError(f"Unknown alignment algorithm: {algorithm}")

    reward_accuracies = (chosen_rewards > rejected_rewards).float().mean().item()
    reward_margin = (chosen_rewards - rejected_rewards).mean().item()

    return losses.mean(), {
        "loss": round(losses.mean().item(), 4),
        "reward_accuracy": round(reward_accuracies, 4),
        "reward_margin": round(reward_margin, 4),
        "chosen_reward": round(chosen_rewards.mean().item(), 4),
        "rejected_reward": round(rejected_rewards.mean().item(), 4),
    }


def _encode_pair(
    prompt: str,
    response: str,
    tokenizer: Any,
    max_seq_length: int,
) -> Tuple[List[int], List[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    resp_ids = tokenizer.encode(response, add_special_tokens=False)
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        resp_ids = resp_ids + [tokenizer.eos_token_id]

    full_ids = (prompt_ids + resp_ids)[:max_seq_length]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return full_ids, labels


def _collate_pref_batch(
    batch: List[Dict[str, str]],
    tokenizer: Any,
    max_seq_length: int = 512,
) -> Dict[str, torch.Tensor]:
    chosen_ids_list, chosen_lbl_list = [], []
    rejected_ids_list, rejected_lbl_list = [], []

    for item in batch:
        c_ids, c_lbl = _encode_pair(item["prompt"], item["chosen"], tokenizer, max_seq_length)
        r_ids, r_lbl = _encode_pair(item["prompt"], item["rejected"], tokenizer, max_seq_length)
        chosen_ids_list.append(c_ids)
        chosen_lbl_list.append(c_lbl)
        rejected_ids_list.append(r_ids)
        rejected_lbl_list.append(r_lbl)

    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def _pad(seqs, lbls):
        m_len = max(len(s) for s in seqs)
        p_ids, p_lbls, masks = [], [], []
        for s, l in zip(seqs, lbls):
            pad_len = m_len - len(s)
            p_ids.append(s + [pad_id] * pad_len)
            p_lbls.append(l + [-100] * pad_len)
            masks.append([1] * len(s) + [0] * pad_len)
        return (
            torch.tensor(p_ids, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
            torch.tensor(p_lbls, dtype=torch.long),
        )

    c_input, c_mask, c_labels = _pad(chosen_ids_list, chosen_lbl_list)
    r_input, r_mask, r_labels = _pad(rejected_ids_list, rejected_lbl_list)

    return {
        "chosen_input_ids": c_input,
        "chosen_attention_mask": c_mask,
        "chosen_labels": c_labels,
        "rejected_input_ids": r_input,
        "rejected_attention_mask": r_mask,
        "rejected_labels": r_labels,
    }


def run_preference_training(
    model: torch.nn.Module,
    tokenizer: Any,
    pref_dataset: Any,
    output_dir: str,
    algorithm: str = "simpo",
    beta: float = 2.0,
    simpo_gamma: float = 0.5,
    learning_rate: float = 5e-6,
    num_epochs: int = 1,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    max_seq_length: int = 512,
    max_steps: int = -1,
) -> Dict[str, Any]:
    """
    执行 Stage 2 偏好对齐训练 (支持 DPO 与 SimPO 一键切换).
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.train()

    # 若算法为 DPO，我们利用 PEFT 的 disable_adapter() 或深拷贝冻结模型作为参考模型 π_ref
    use_peft_disable = hasattr(model, "disable_adapter")
    ref_model = None
    if algorithm.lower() == "dpo" and not use_peft_disable:
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    dataloader = DataLoader(
        pref_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate_pref_batch(b, tokenizer, max_seq_length),
    )

    history: List[Dict[str, float]] = []
    global_step = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # 1. 策略模型 π_theta 前向计算
            chosen_out = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )
            rejected_out = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )

            pol_c_sum, pol_c_avg = get_sequence_logps(chosen_out.logits, batch["chosen_labels"])
            pol_r_sum, pol_r_avg = get_sequence_logps(rejected_out.logits, batch["rejected_labels"])

            # 2. 若为 DPO，计算参考模型 π_ref 的对数似然（SimPO 直接跳过此步，节省算力！）
            ref_c_sum, ref_r_sum = None, None
            if algorithm.lower() == "dpo":
                with torch.no_grad():
                    if use_peft_disable:
                        with model.disable_adapter():
                            ref_c_out = model(
                                input_ids=batch["chosen_input_ids"],
                                attention_mask=batch["chosen_attention_mask"],
                            )
                            ref_r_out = model(
                                input_ids=batch["rejected_input_ids"],
                                attention_mask=batch["rejected_attention_mask"],
                            )
                    else:
                        ref_c_out = ref_model(
                            input_ids=batch["chosen_input_ids"],
                            attention_mask=batch["chosen_attention_mask"],
                        )
                        ref_r_out = ref_model(
                            input_ids=batch["rejected_input_ids"],
                            attention_mask=batch["rejected_attention_mask"],
                        )
                    ref_c_sum, _ = get_sequence_logps(ref_c_out.logits, batch["chosen_labels"])
                    ref_r_sum, _ = get_sequence_logps(ref_r_out.logits, batch["rejected_labels"])

            # 3. 计算 DPO / SimPO Loss
            loss, step_stats = compute_dpo_or_simpo_loss(
                policy_chosen_sum_logps=pol_c_sum,
                policy_chosen_avg_logps=pol_c_avg,
                policy_rejected_sum_logps=pol_r_sum,
                policy_rejected_avg_logps=pol_r_avg,
                ref_chosen_sum_logps=ref_c_sum,
                ref_rejected_sum_logps=ref_r_sum,
                algorithm=algorithm,
                beta=beta,
                simpo_gamma=simpo_gamma,
            )

            (loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                step_stats["step"] = global_step
                history.append(step_stats)

                if max_steps > 0 and global_step >= max_steps:
                    break
        if max_steps > 0 and global_step >= max_steps:
            break

    elapsed = round(time.time() - start_time, 2)
    adapter_save_path = os.path.join(output_dir, f"final_{algorithm.lower()}_adapter")
    os.makedirs(adapter_save_path, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(adapter_save_path)

    summary = {
        "stage": "preference_alignment",
        "algorithm": algorithm.lower(),
        "beta": beta,
        "simpo_gamma": simpo_gamma if algorithm.lower() == "simpo" else None,
        "total_steps": global_step,
        "final_metrics": history[-1] if history else {},
        "elapsed_sec": elapsed,
        "history": history,
        "adapter_path": adapter_save_path,
    }
    with open(
        os.path.join(output_dir, f"{algorithm.lower()}_training_log.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
