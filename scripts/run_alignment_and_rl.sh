#!/usr/bin/env bash
# ============================================================================
# 实验二 (RQ2) & 实验三 (RQ3)：运行 DPO vs. SimPO 偏好对齐与 GRPO 强化学习并执行多维评测
# ============================================================================
set -e

echo "=== [Stage 2 - DPO] 有参考模型直接偏好优化 ==="
python run_pipeline.py --mode align --config configs/align_dpo_simpo.yaml --algo dpo

echo "=== [Stage 2 - SimPO] 无参考模型 + 长度归一化偏好优化 ==="
python run_pipeline.py --mode align --config configs/align_dpo_simpo.yaml --algo simpo

echo "=== [Stage 3 - GRPO] 基于规则奖励的群组相对策略优化 ==="
python run_pipeline.py --mode grpo --config configs/rl_grpo.yaml

echo "=== [Evaluation] 域内 GSM8K 与 域外 SVAMP 泛化评测 ==="
python run_pipeline.py --mode eval --benchmark gsm8k --adapter_path outputs/stage3_grpo/final_grpo_adapter
python run_pipeline.py --mode eval --benchmark svamp --adapter_path outputs/stage3_grpo/final_grpo_adapter

echo "=== [Visualization] 生成科研论文高清图表 ==="
python run_pipeline.py --mode plot
