"""
Stage 1: 监督微调 (Supervised Fine-Tuning, SFT) 训练器.

科研教学要点：
1. 在指令微调中，输入由 Prompt (问题 $x$) 和 Completion (包含 <think> 推导过程与 <answer> 的解答 $y$) 拼接而成。
2. 核心技巧 — **Prompt Masking (标签掩码)**：
   我们在计算交叉熵损失 $\\mathcal{L}_{\\text{SFT}} = -\\sum_{t=1}^{|y|} \\log P_\\theta(y_t | x, y_{<t})$ 时，
   会将 Prompt 部分的 `labels` 设为 `-100`（PyTorch CrossEntropyLoss 默认忽略 -100），
   使模型专注于学习“如何推导和回答”，而不是死记硬背题目本身。
"""

import os
import json
import time
from typing import Dict, Any, List
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW


def _collate_sft_batch(
    batch: List[Dict[str, str]],
    tokenizer: Any,
    max_seq_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """
    将一अच批 (prompt, completion) 样本编码为 input_ids, attention_mask, labels，
    并对 Prompt 部分施加 -100 掩码.
    """
    input_ids_list = []
    labels_list = []

    for item in batch:
        prompt_ids = tokenizer.encode(item["prompt"], add_special_tokens=True)
        comp_ids = tokenizer.encode(item["completion"], add_special_tokens=False)
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            comp_ids = comp_ids + [tokenizer.eos_token_id]

        full_ids = (prompt_ids + comp_ids)[:max_seq_length]
        # Prompt 部分 label 设为 -100，仅对 completion 部分计算梯度
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        input_ids_list.append(full_ids)
        labels_list.append(labels)

    max_len = max(len(x) for x in input_ids_list)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    padded_inputs = []
    padded_labels = []
    attention_masks = []
    for ids, lbls in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        padded_inputs.append(ids + [pad_id] * pad_len)
        padded_labels.append(lbls + [-100] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def run_sft_training(
    model: torch.nn.Module,
    tokenizer: Any,
    train_dataset: Any,
    output_dir: str,
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    max_seq_length: int = 512,
    max_steps: int = -1,
) -> Dict[str, Any]:
    """
    执行 Stage 1 SFT 微调，记录步级 Loss 曲线与显存开销，并保存 LoRA/DoRA 权重.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.train()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate_sft_batch(b, tokenizer, max_seq_length),
    )

    loss_history: List[Dict[str, float]] = []
    global_step = 0
    running_loss = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            running_loss += outputs.loss.item()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                avg_step_loss = running_loss / grad_accum_steps
                running_loss = 0.0
                loss_history.append(
                    {
                        "step": global_step,
                        "epoch": round(epoch + (step + 1) / len(dataloader), 3),
                        "loss": round(avg_step_loss, 4),
                    }
                )

                if max_steps > 0 and global_step >= max_steps:
                    break
        if max_steps > 0 and global_step >= max_steps:
            break

    elapsed = round(time.time() - start_time, 2)
    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if torch.cuda.is_available()
        else 0.0
    )

    adapter_save_path = os.path.join(output_dir, "final_adapter")
    os.makedirs(adapter_save_path, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(adapter_save_path)

    metrics = {
        "stage": "sft",
        "total_steps": global_step,
        "final_loss": loss_history[-1]["loss"] if loss_history else None,
        "elapsed_sec": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "loss_history": loss_history,
        "adapter_path": adapter_save_path,
    }
    with open(os.path.join(output_dir, "sft_training_log.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics
