#!/usr/bin/env bash
# ============================================================================
# 实验一 (RQ1 消融实验)：对比 LoRA vs. DoRA 在不同秩 (r=8, 16, 64) 与目标模块下的表现
# 适用环境：Google Colab (T4 / L4) 或 AWS (g4dn.xlarge / g5.xlarge)
# ============================================================================
set -e

echo "=== [RQ1 Ablation 1/4] LoRA (r=16, attn_only) ==="
python run_pipeline.py --mode sft --config configs/sft_lora.yaml --lora_r 16 --target_scope attn_only

echo "=== [RQ1 Ablation 2/4] LoRA (r=16, all_linear) ==="
python run_pipeline.py --mode sft --config configs/sft_lora.yaml --lora_r 16 --target_scope all_linear

echo "=== [RQ1 Ablation 3/4] DoRA (r=16, attn_only) ==="
python run_pipeline.py --mode sft --config configs/sft_lora.yaml --use_dora --lora_r 16 --target_scope attn_only

echo "=== [RQ1 Ablation 4/4] DoRA (r=16, all_linear) ==="
python run_pipeline.py --mode sft --config configs/sft_lora.yaml --use_dora --lora_r 16 --target_scope all_linear
