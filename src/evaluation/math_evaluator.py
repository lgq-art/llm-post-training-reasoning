"""
多维度数学推理评测器 (Math Reasoning Evaluator).

科研评测体系设计：
在顶会论文（如 DeepSeekMath, Qwen2.5-Math, SimPO）中，仅汇报单一 Greedy Accuracy 是不够的。
本评测器同时计算以下核心科研指标：
1. **Pass@1 (Greedy)**: 温度 T=0 确定性解码下的精确匹配率（反映模型单次推理稳定性）。
2. **Pass@k (Oracle Coverage)**: 采样 k 条轨迹中至少有 1 条正确的比例（反映模型潜在知识上限，以及 RL 能否突破上限）。
3. **Maj@k (Self-Consistency / Majority Voting)**: 采样 k 条轨迹进行多数投票后的准确率（Wang et al., ICLR 2023）。
4. **Avg. Reasoning Tokens & Format Compliance (%)**: 统计平均生成 Token 长度与 `<think>...</think><answer>\\boxed{}</answer>` 严格遵循率，用于回答 RQ2（长度偏见）与 RQ3（格式涌现）。
"""

import os
import json
from collections import Counter
from typing import Dict, Any, List, Optional
import torch
from ..data.prompt_templates import (
    extract_predicted_answer,
    is_answer_correct,
    check_format_compliance,
)


def compute_pass_and_maj_at_k(
    sampled_outputs: List[str],
    gold_answer: str,
) -> Dict[str, Any]:
    """
    针对单道题目采样得到的 k 条输出，计算：
    - pass_at_k: 是否有至少 1 条预测正确
    - maj_at_k: 多数投票选出的答案是否正确
    """
    extracted_preds: List[str] = []
    any_correct = False

    for out in sampled_outputs:
        pred = extract_predicted_answer(out)
        if pred is not None:
            extracted_preds.append(pred)
            if is_answer_correct(pred, gold_answer):
                any_correct = True

    if not extracted_preds:
        maj_pred = None
        maj_correct = False
    else:
        maj_pred = Counter(extracted_preds).most_common(1)[0][0]
        maj_correct = is_answer_correct(maj_pred, gold_answer)

    return {
        "pass_at_k": 1.0 if any_correct else 0.0,
        "maj_at_k": 1.0 if maj_correct else 0.0,
        "majority_prediction": maj_pred,
        "all_predictions": extracted_preds,
    }


def evaluate_math_benchmark(
    model: torch.nn.Module,
    tokenizer: Any,
    eval_dataset: Any,
    output_dir: str,
    benchmark_name: str = "gsm8k",
    num_samples_k: int = 4,
    max_new_tokens: int = 256,
    temperature_k: float = 0.7,
) -> Dict[str, Any]:
    """
    在指定基准数据集（如 GSM8K 或 SVAMP）上运行完整评测并导出详细案例分析 JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    records: List[Dict[str, Any]] = []
    pass1_hits = 0
    passk_hits = 0.0
    majk_hits = 0.0
    format_hits = 0
    total_tokens = 0

    for idx, item in enumerate(eval_dataset):
        prompt = item["prompt"]
        gold = item["gold_answer"]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_tensor)
        prompt_len = input_tensor.shape[1]

        with torch.no_grad():
            # 1. Greedy Decode (Pass@1)
            greedy_seq = model.generate(
                input_ids=input_tensor,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
            greedy_comp_ids = greedy_seq[0, prompt_len:]
            greedy_text = tokenizer.decode(greedy_comp_ids, skip_special_tokens=True)

            # 2. Sample k Trajectories (Pass@k & Maj@k)
            if num_samples_k > 1:
                batch_input = input_tensor.repeat(num_samples_k, 1)
                batch_mask = attn_mask.repeat(num_samples_k, 1)
                sampled_seqs = model.generate(
                    input_ids=batch_input,
                    attention_mask=batch_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature_k,
                    top_p=0.95,
                    pad_token_id=pad_id,
                )
                sampled_texts = tokenizer.batch_decode(
                    sampled_seqs[:, prompt_len:], skip_special_tokens=True
                )
            else:
                sampled_texts = [greedy_text]

        pred_1 = extract_predicted_answer(greedy_text)
        correct_1 = is_answer_correct(pred_1, gold)
        fmt_info = check_format_compliance(greedy_text)
        k_stats = compute_pass_and_maj_at_k(sampled_texts, gold)

        gen_token_len = int((greedy_comp_ids != pad_id).sum().item())
        pass1_hits += 1 if correct_1 else 0
        passk_hits += k_stats["pass_at_k"]
        majk_hits += k_stats["maj_at_k"]
        format_hits += 1 if (fmt_info["has_think"] and fmt_info["has_boxed"]) else 0
        total_tokens += gen_token_len

        records.append(
            {
                "id": idx,
                "question": item["question"],
                "gold_answer": gold,
                "greedy_prediction": pred_1,
                "greedy_correct": correct_1,
                "pass_at_k": bool(k_stats["pass_at_k"]),
                "maj_at_k": bool(k_stats["maj_at_k"]),
                "generated_tokens": gen_token_len,
                "format_compliant": fmt_info["has_think"] and fmt_info["has_boxed"],
                "greedy_output": greedy_text,
            }
        )

    n = max(len(records), 1)
    pass1_acc = round(100.0 * pass1_hits / n, 2)
    passk_acc = round(100.0 * passk_hits / n, 2)
    majk_acc = round(100.0 * majk_hits / n, 2)
    fmt_rate = round(100.0 * format_hits / n, 2)
    avg_tokens = round(total_tokens / n, 1)

    summary = {
        "benchmark": benchmark_name,
        "num_evaluated": len(records),
        "k": num_samples_k,
        "pass_at_1": pass1_acc,
        "pass_at_k": passk_acc,
        "maj_at_k": majk_acc,
        "format_compliance_rate": fmt_rate,
        "avg_generated_tokens": avg_tokens,
        "token_efficiency_score": round(pass1_acc / max(avg_tokens / 100.0, 0.1), 2),
    }

    out_file = os.path.join(output_dir, f"eval_{benchmark_name}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records[:50]}, f, indent=2, ensure_ascii=False)

    return summary
