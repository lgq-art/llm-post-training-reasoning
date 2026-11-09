# 小尺寸大模型复杂推理能力的后训练与强化对齐研究
**(Post-Training & Reinforcement Alignment for Small LLM Mathematical Reasoning: From SFT/DoRA and DPO/SimPO to GRPO)**

本项目是一个专为**本科生科研训练与大模型前沿实践**设计的标准研究型代码库。不依赖黑盒 API 套壳，而是从零实现大模型后训练（Post-Training）的三大核心阶段，并围绕三个明确的**科学问题（Research Questions, RQs）**展开消融实验（Ablation Study）与机理分析。

---

## 🧭 第一次做大模型科研实践：标准操作步骤 (SOP 手把手指南)

在正式科研中，我们遵循 **“本机秒级联调（Smoke Test） $\rightarrow$ 云端 GPU 小步快跑验证 $\rightarrow$ 批量消融实验 $\rightarrow$ 多维指标评测与出图”** 的四步走标准流程，绝不盲目在昂贵的 GPU 上直接跑未经验证的代码。

### 第一步：在本机运行 30 秒“冒烟测试” (Smoke Test)
在把代码上传到 Google Colab 或 AWS 之前，先在您自己的电脑上运行内置的离线冒烟测试。该模式会自动构建一个微型 Qwen2 模型与合成数学推理数据集，在不消耗 GPU 算力、不下载数 GB 权重的情况下，验证 **SFT $\rightarrow$ DPO/SimPO $\rightarrow$ GRPO $\rightarrow$ 多维评测 $\rightarrow$ 论文图表绘制** 全链路 100% 畅通：

```bash
python run_pipeline.py --mode smoke_test
```
运行成功后，您会在 `outputs/figures/` 目录下直接看到生成的 4 张高清科研图表，并在 `outputs/smoke_test/` 下看到各个阶段的训练日志 JSON。

---

### 第二步：在 Google Colab 或 AWS 学生算力平台部署环境

#### 方案 A：使用 Google Colab（推荐新手首选，零配置）
1. 打开 [Google Colab](https://colab.research.google.com/)，点击顶部菜单 **`代码执行程序 (Runtime)` $\rightarrow$ `更改运行时类型` $\rightarrow$ 选择 `T4 GPU`（或 `L4 GPU`）**。
2. 将本项目文件夹上传至您的 Google Drive（或推送到 GitHub 仓库后在 Colab 中 `!git clone`）。
3. 在 Colab 中打开 [`notebooks/Colab_AWS_实战演练手册.ipynb`](./notebooks/Colab_AWS_实战演练手册.ipynb)，按照单元格提示依次点击运行即可。

#### 方案 B：使用 AWS Educate / Academy 学生算力
1. 启动一台配备单张 GPU 的 EC2 实例（推荐 `g4dn.xlarge` [NVIDIA T4 16GB] 或 `g5.xlarge` [NVIDIA A10G 24GB]，镜像选择 **Deep Learning OSS Nvidia Driver AMI GPU PyTorch**）。
2. 通过 SSH 登录实例后，克隆本项目并安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

---

### 第三步：按“控制变量法”执行三阶段科研实验

#### 🔬 实验一（回答 RQ1）：参数高效微调结构消融（LoRA vs. DoRA & 插入位置）
探究低秩适配器的权重分解（DoRA）以及在“仅注意力层 (`attn_only`)”与“全线性层 (`all_linear`, 含 MLP)”注入对数学思维链拟合能力的影响：

```bash
# 1. 标准 QLoRA (r=16, 仅 Attention 层)
python run_pipeline.py --mode sft --config configs/sft_lora.yaml --lora_r 16 --target_scope attn_only

# 2. 标准 QLoRA (r=16, Attention + MLP 全线性层)
python run_pipeline.py --mode sft --config configs/sft_lora.yaml --lora_r 16 --target_scope all_linear

# 3. 权重分解 DoRA (r=16, Attention + MLP 全线性层)
python run_pipeline.py --mode sft --config configs/sft_lora.yaml --use_dora --lora_r 16 --target_scope all_linear
```
> **新手提示**：第一次在 GPU 上跑时，可以在命令末尾加上 `--max_steps 100`（约 5 分钟跑完），先观察 `outputs/stage1_sft/sft_training_log.json` 中的 `loss` 是否平稳下降，确认无误后再去掉 `--max_steps` 跑完整数据集。

#### 🔬 实验二（回答 RQ2）：离线偏好对齐算法对比（DPO vs. SimPO）
基于 Stage 1 训练好的 SFT 模型，对比**有参考模型的 DPO** 与**无参考模型且引入长度归一化的 SimPO**，观察谁能更好地抑制“冗长废话（Length Bias）”并节省显存：

```bash
# 1. 运行 DPO (β = 0.1, 自动使用 PEFT disable_adapter 作为冻结参考模型 π_ref)
python run_pipeline.py --mode align --config configs/align_dpo_simpo.yaml --algo dpo

# 2. 运行 SimPO (β = 2.0, γ = 0.5, 无需参考模型，按序列长度归一化)
python run_pipeline.py --mode align --config configs/align_dpo_simpo.yaml --algo simpo
```

#### 🔬 实验三（回答 RQ3）：在线规则强化学习（GRPO, DeepSeek-R1 范式）
无需价值网络（Critic Model），对每个数学问题在线采样 $G=4$ 条不同轨迹，使用确定性规则（答案数值精确匹配 + `<think>...</think><answer>\boxed{}</answer>` 格式遵循度）计算组内相对优势 $A_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}$：

```bash
python run_pipeline.py --mode grpo --config configs/rl_grpo.yaml
```

---

### 第四步：运行多维基准评测与生成论文级图表

分别在 **域内测试集 (`gsm8k`)** 与 **域外泛化测试集 (`svamp`)** 上评测模型的 `Pass@1 (Greedy)`、`Pass@8`、`Maj@8 (Self-Consistency)`、`平均思考 Token 长度` 与 `格式遵循率`：

```bash
# 评测域内 GSM8K
python run_pipeline.py --mode eval --benchmark gsm8k --adapter_path outputs/stage3_grpo/final_grpo_adapter

# 评测域外 OOD 泛化基准 SVAMP
python run_pipeline.py --mode eval --benchmark svamp --adapter_path outputs/stage3_grpo/final_grpo_adapter

# 一键生成 4 张 300 DPI 科研论文图表
python run_pipeline.py --mode plot
```

---

## 📖 代码模块与核心数学原理对照表（边看代码边学理论）

| 核心模块文件 | 对应科研概念与算法 | 关键实现函数 |
| :--- | :--- | :--- |
| [`src/data/prompt_templates.py`](./src/data/prompt_templates.py) | 结构化思维链 `<think>` 模板与答案正则提取 | `format_sft_example()`, `extract_predicted_answer()`, `check_format_compliance()` |
| [`src/models/model_factory.py`](./src/models/model_factory.py) | 4-bit NF4 QLoRA 量化与 LoRA/DoRA 权重分解注入 | `build_peft_config()`, `load_model_and_tokenizer()` |
| [`src/trainers/sft_trainer.py`](./src/trainers/sft_trainer.py) | **Prompt Masking (`labels = -100`)**：仅对解答部分计算交叉熵损失 | `_collate_sft_batch()`, `run_sft_training()` |
| [`src/trainers/pref_trainer.py`](./src/trainers/pref_trainer.py) | **DPO vs. SimPO 数学公式对比实现**：序列总对数似然 vs. 长度平均对数似然 | `get_sequence_logps()`, `compute_dpo_or_simpo_loss()` |
| [`src/trainers/grpo_trainer.py`](./src/trainers/grpo_trainer.py) | **GRPO 群组相对策略优化**：群组采样、可验证规则奖励、Schulman 无偏 KL 惩罚 | `compute_rule_rewards()`, `compute_group_advantages()`, `run_grpo_training()` |
| [`src/evaluation/math_evaluator.py`](./src/evaluation/math_evaluator.py) | **Pass@1, Pass@k 与 Maj@k (多数投票自洽性)** 评测体系 | `compute_pass_and_maj_at_k()`, `evaluate_math_benchmark()` |
| [`src/visualization/plot_results.py`](./src/visualization/plot_results.py) | 顶会风格实验图表绘制（主实验对比、消融实验、长度分布、RL 收敛曲线） | `generate_all_research_figures()` |
