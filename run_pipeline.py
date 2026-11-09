"""
大模型复杂推理后训练与强化对齐 — 统一实验入口 (Unified Research Pipeline CLI).

使用方法速览：
1. 本机 30 秒全链路冒烟测试（无需 GPU、无需联网下载权重，验证所有代码与数学公式）：
   python run_pipeline.py --mode smoke_test

2. 云端 GPU (Colab T4 / AWS g4dn/g5) 正式科研实验：
   # 阶段一：监督微调 (SFT with LoRA / DoRA)
   python run_pipeline.py --mode sft --config configs/sft_lora.yaml
   python run_pipeline.py --mode sft --config configs/sft_lora.yaml --use_dora --lora_r 16

   # 阶段二：离线偏好对齐 (DPO vs. SimPO)
   python run_pipeline.py --mode align --config configs/align_dpo_simpo.yaml --algo dpo
   python run_pipeline.py --mode align --config configs/align_dpo_simpo.yaml --algo simpo

   # 阶段三：在线规则强化学习 (GRPO)
   python run_pipeline.py --mode grpo --config configs/rl_grpo.yaml

   # 多维基准评测 (GSM8K 域内 + SVAMP 域外泛化)
   python run_pipeline.py --mode eval --benchmark gsm8k --adapter_path outputs/stage3_grpo/final_grpo_adapter

   # 生成全套科研论文图表
   python run_pipeline.py --mode plot
"""

import os
import sys
import argparse
import random
import yaml
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data import (
    load_sft_dataset,
    load_preference_dataset,
    load_grpo_dataset,
    load_eval_dataset,
)
from src.models import (
    load_model_and_tokenizer,
    build_peft_config,
    get_trainable_param_stats,
)
from src.trainers import (
    run_sft_training,
    run_preference_training,
    run_grpo_training,
)
from src.evaluation import evaluate_math_benchmark
from src.visualization import generate_all_research_figures


def set_Global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_smoke_test() -> None:
    """
    本机秒级全流程冒烟测试：
    依次验证 RQ1 (LoRA/DoRA 注入)、Stage 1 (SFT)、Stage 2 (DPO & SimPO)、
    Stage 3 (GRPO 规则强化学习)、多维评测器 (Pass@1/Maj@k) 以及科研图表生成。
    """
    print("=" * 75)
    print("🚀 [Smoke Test] 启动大模型后训练全链路本机验证 (无需 GPU / 离线可运行)")
    print("=" * 75)
    set_Global_seed(42)

    # 1. 验证 RQ1 不同 PEFT 配置的参数量统计
    print("\n[Step 1/6] 验证 RQ1: LoRA vs. DoRA 及不同注入范围 (Attn-Only vs. All-Linear)...")
    for scope in ["attn_only", "all_linear"]:
        for use_dora in [False, True]:
            peft_cfg = build_peft_config(
                lora_r=8, lora_alpha=16, use_dora=use_dora, target_scope=scope
            )
            model, tokenizer = load_model_and_tokenizer(
                peft_config=peft_cfg, use_tiny_mock=True
            )
            stats = get_trainable_param_stats(model)
            tag = "DoRA" if use_dora else "LoRA"
            print(
                f"  -> {tag} ({scope:10s}): 可训练参数 = {stats['trainable_params']:,} "
                f"/ 总参数 = {stats['all_params']:,} ({stats['trainable_percentage']:.2f}%)"
            )

    # 2. 验证 Stage 1: SFT 监督微调
    print("\n[Step 2/6] 验证 Stage 1: SFT 监督微调 (含 Prompt Masking)...")
    sft_ds = load_sft_dataset(max_samples=16, use_synthetic=True)
    sft_res = run_sft_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_ds,
        output_dir="./outputs/smoke_test/stage1_sft",
        learning_rate=1e-3,
        num_epochs=2,
        batch_size=4,
        grad_accum_steps=1,
        max_seq_length=256,
    )
    losses = [x["loss"] for x in sft_res["loss_history"]]
    print(f"  -> SFT 训练完成! 初始 Loss = {losses[0]:.4f} -> 最终 Loss = {losses[-1]:.4f}")

    # 3. 验证 Stage 2: DPO vs. SimPO 离线偏好对齐
    print("\n[Step 3/6] 验证 Stage 2: DPO 与 SimPO 偏好对齐算法...")
    pref_ds = load_preference_dataset(max_samples=12, use_synthetic=True)
    for algo, beta in [("dpo", 0.1), ("simpo", 2.0)]:
        pref_res = run_preference_training(
            model=model,
            tokenizer=tokenizer,
            pref_dataset=pref_ds,
            output_dir=f"./outputs/smoke_test/stage2_{algo}",
            algorithm=algo,
            beta=beta,
            simpo_gamma=0.5,
            learning_rate=5e-4,
            num_epochs=1,
            batch_size=4,
            grad_accum_steps=1,
            max_seq_length=256,
        )
        m = pref_res["final_metrics"]
        print(
            f"  -> [{algo.upper():5s}] Loss = {m['loss']:.4f}, "
            f"Reward Acc = {m['reward_accuracy']*100:.1f}%, Margin = {m['reward_margin']:.4f}"
        )

    # 4. 验证 Stage 3: GRPO 群组相对策略优化强化学习
    print("\n[Step 4/6] 验证 Stage 3: GRPO 在线采样与可验证规则奖励强化学习...")
    grpo_ds = load_grpo_dataset(max_samples=6, use_synthetic=True)
    grpo_res = run_grpo_training(
        model=model,
        tokenizer=tokenizer,
        grpo_dataset=grpo_ds,
        output_dir="./outputs/smoke_test/stage3_grpo",
        group_size=4,
        temperature=0.8,
        kl_coef=0.04,
        learning_rate=5e-4,
        max_prompt_length=128,
        max_completion_length=32,
        max_steps=4,
    )
    gm = grpo_res["final_metrics"]
    print(
        f"  -> [GRPO] 完成 {grpo_res['total_steps']} 步群组采样更新! "
        f"最终 Loss = {gm['loss']:.4f}, KL 散度 = {gm['kl_div']:.5f}"
    )

    # 5. 验证多维数学推理评测器
    print("\n[Step 5/6] 验证多维数学推理评测器 (Pass@1, Pass@k, Maj@k, Token 长度统计)...")
    eval_ds = load_eval_dataset(benchmark="gsm8k", max_samples=6, use_synthetic=True)
    eval_summary = evaluate_math_benchmark(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_ds,
        output_dir="./outputs/smoke_test/eval",
        benchmark_name="gsm8k_synthetic",
        num_samples_k=4,
        max_new_tokens=32,
    )
    print(f"  -> 评测结果摘要: {eval_summary}")

    # 6. 生成全套科研级对比图表
    print("\n[Step 6/6] 生成 4 张高清科研报告对比图表...")
    fig_paths = generate_all_research_figures("./outputs/figures")
    for p in fig_paths:
        print(f"  -> 已生成图表: {p}")

    print("\n" + "=" * 75)
    print("✅ 全链路冒烟测试 100% 通过！代码、公式与评测绘图管线均已就绪。")
    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Reasoning Post-Training Research Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="smoke_test",
        choices=["smoke_test", "sft", "align", "grpo", "eval", "plot"],
        help="运行模式：smoke_test | sft | align | grpo | eval | plot",
    )
    parser.add_argument("--config", type=str, default="configs/sft_lora.yaml", help="YAML 配置文件路径")
    parser.add_argument("--use_dora", action="store_true", help="RQ1 消融：启用 DoRA 替代标准 LoRA")
    parser.add_argument("--lora_r", type=int, default=None, help="RQ1 消融：覆盖配置文件中的 lora_r")
    parser.add_argument("--target_scope", type=str, default=None, choices=["attn_only", "all_linear"])
    parser.add_argument("--algo", type=str, default=None, choices=["dpo", "simpo"], help="RQ2 消融：选择 dpo 或 simpo")
    parser.add_argument("--benchmark", type=str, default="gsm8k", choices=["gsm8k", "svamp"])
    parser.add_argument("--adapter_path", type=str, default=None, help="评测或继续训练时加载的 LoRA 权重路径")
    parser.add_argument("--max_steps", type=int, default=-1, help="限制最大训练步数（用于快速调试）")
    args = parser.parse_args()

    if args.mode == "smoke_test":
        run_smoke_test()
        return

    if args.mode == "plot":
        paths = generate_all_research_figures("./outputs/figures")
        for p in paths:
            print(f"Generated figure: {p}")
        return

    cfg = load_yaml_config(args.config)
    set_Global_seed(cfg.get("seed", 42))

    if args.mode == "sft":
        peft_sec = cfg["peft"]
        use_dora = True if args.use_dora else peft_sec.get("use_dora", False)
        lora_r = args.lora_r if args.lora_r is not None else peft_sec.get("lora_r", 16)
        scope = args.target_scope if args.target_scope is not None else peft_sec.get("target_scope", "all_linear")

        peft_cfg = build_peft_config(
            lora_r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=peft_sec.get("lora_dropout", 0.05),
            use_dora=use_dora,
            target_scope=scope,
        )
        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path=cfg["model"]["model_name_or_path"],
            use_4bit=cfg["model"].get("use_4bit_quantization", True),
            peft_config=peft_cfg,
        )
        print("Trainable parameter stats:", get_trainable_param_stats(model))

        train_ds = load_sft_dataset(
            dataset_name=cfg["data"]["dataset_name"],
            subset=cfg["data"].get("dataset_subset", "main"),
            max_samples=cfg["data"].get("max_train_samples", -1),
        )
        t_cfg = cfg["training"]
        res = run_sft_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            output_dir=t_cfg["output_dir"],
            learning_rate=float(t_cfg["learning_rate"]),
            num_epochs=t_cfg["num_train_epochs"],
            batch_size=t_cfg["per_device_train_batch_size"],
            grad_accum_steps=t_cfg["gradient_accumulation_steps"],
            max_seq_length=cfg["model"].get("max_seq_length", 1024),
            max_steps=args.max_steps,
        )
        print(f"Stage 1 SFT finished. Final loss: {res['final_loss']}, saved to {res['adapter_path']}")

    elif args.mode == "align":
        algo = args.algo if args.algo else cfg["alignment"]["algorithm"]
        beta = 0.1 if algo == "dpo" and args.algo == "dpo" else float(cfg["alignment"]["beta"])
        peft_sec = cfg["peft"]
        peft_cfg = build_peft_config(
            lora_r=peft_sec.get("lora_r", 16),
            lora_alpha=peft_sec.get("lora_alpha", 32),
            target_scope=peft_sec.get("target_scope", "all_linear"),
        )
        sft_adapter = args.adapter_path or cfg["model"].get("sft_adapter_path")
        adapter_to_load = sft_adapter if (sft_adapter and os.path.exists(sft_adapter)) else None

        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path=cfg["model"]["base_model_name_or_path"],
            use_4bit=cfg["model"].get("use_4bit_quantization", True),
            peft_config=peft_cfg if not adapter_to_load else None,
            adapter_path=adapter_to_load,
        )
        pref_ds = load_preference_dataset(
            dataset_name=cfg["data"]["dataset_name"],
            max_samples=cfg["data"].get("max_train_samples", 2000),
        )
        t_cfg = cfg["training"]
        res = run_preference_training(
            model=model,
            tokenizer=tokenizer,
            pref_dataset=pref_ds,
            output_dir=t_cfg["output_dir"],
            algorithm=algo,
            beta=beta,
            simpo_gamma=float(cfg["alignment"].get("simpo_gamma", 0.5)),
            learning_rate=float(t_cfg["learning_rate"]),
            num_epochs=t_cfg["num_train_epochs"],
            batch_size=t_cfg["per_device_train_batch_size"],
            grad_accum_steps=t_cfg["gradient_accumulation_steps"],
            max_seq_length=cfg["model"].get("max_seq_length", 1024),
            max_steps=args.max_steps,
        )
        print(f"Stage 2 ({algo.upper()}) finished. Saved to {res['adapter_path']}")

    elif args.mode == "grpo":
        peft_sec = cfg["peft"]
        peft_cfg = build_peft_config(
            lora_r=peft_sec.get("lora_r", 16),
            lora_alpha=peft_sec.get("lora_alpha", 32),
            target_scope=peft_sec.get("target_scope", "all_linear"),
        )
        sft_adapter = args.adapter_path or cfg["model"].get("sft_adapter_path")
        adapter_to_load = sft_adapter if (sft_adapter and os.path.exists(sft_adapter)) else None

        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path=cfg["model"]["model_name_or_path"],
            use_4bit=cfg["model"].get("use_4bit_quantization", True),
            peft_config=peft_cfg if not adapter_to_load else None,
            adapter_path=adapter_to_load,
        )
        grpo_ds = load_grpo_dataset(
            dataset_name=cfg["data"]["dataset_name"],
            subset=cfg["data"].get("dataset_subset", "main"),
            max_samples=cfg["data"].get("max_train_samples", 1500),
        )
        g_cfg = cfg["grpo"]
        t_cfg = cfg["training"]
        res = run_grpo_training(
            model=model,
            tokenizer=tokenizer,
            grpo_dataset=grpo_ds,
            output_dir=t_cfg["output_dir"],
            group_size=g_cfg.get("num_generations", 4),
            temperature=float(g_cfg.get("temperature", 0.8)),
            kl_coef=float(g_cfg.get("kl_coef", 0.04)),
            learning_rate=float(t_cfg["learning_rate"]),
            max_prompt_length=cfg["model"].get("max_prompt_length", 256),
            max_completion_length=cfg["model"].get("max_completion_length", 512),
            max_steps=args.max_steps if args.max_steps > 0 else 200,
        )
        print(f"Stage 3 GRPO finished. Saved to {res['adapter_path']}")

    elif args.mode == "eval":
        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path=cfg["model"].get("model_name_or_path", "Qwen/Qwen2.5-1.5B"),
            use_4bit=True,
            adapter_path=args.adapter_path,
        )
        eval_ds = load_eval_dataset(benchmark=args.benchmark, max_samples=200)
        summary = evaluate_math_benchmark(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_ds,
            output_dir="./outputs/eval_results",
            benchmark_name=args.benchmark,
            num_samples_k=8,
            max_new_tokens=512,
        )
        print("Evaluation Summary:", summary)


if __name__ == "__main__":
    main()
