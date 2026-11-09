"""
数据集加载器：支持 HuggingFace 在线数据集与本地离线合成测试集 (Smoke Test).

设计亮点：
针对初次上手实践的同学，提供了 `use_synthetic=True` 模式。
在没有 GPU 或未安装 HuggingFace `datasets` 库的本地电脑上，可以秒级生成高仿真的数学推理
SFT、DPO 偏好对与 GRPO 数据集（自动兼容 `torch.utils.data.Dataset`），
用于在上传 Colab / AWS 之前 100% 跑通并验证所有代码逻辑。
"""

from typing import List, Dict, Any
from .prompt_templates import (
    format_sft_example,
    format_inference_prompt,
    extract_gold_answer,
)


class SimpleListDataset:
    """轻量级列表数据集封装，兼容 HuggingFace Dataset 与 PyTorch DataLoader."""

    def __init__(self, data: List[Dict[str, Any]]):
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)


def _build_synthetic_math_samples(num_samples: int = 32) -> List[Dict[str, str]]:
    """
    生成用于本地快速联调 (Smoke Test) 的合成数学应用题样本.
    包含严谨的多步推导链、正样本 (chosen) 与带有冗长废话或计算错误的负样本 (rejected).
    """
    samples = []
    for i in range(1, num_samples + 1):
        apples = 3 + (i % 7)
        price_per_apple = 2 + (i % 4)
        discount = 1 + (i % 3)
        total = apples * price_per_apple - discount

        question = (
            f"Alice buys {apples} apples at ${price_per_apple} each. "
            f"She uses a coupon that gives her ${discount} off the total bill. "
            f"How many dollars does Alice pay in total?"
        )
        raw_gsm8k_answer = (
            f"First, calculate the cost of {apples} apples before the coupon: "
            f"{apples} * {price_per_apple} = {apples * price_per_apple}.\n"
            f"Next, subtract the ${discount} coupon discount from the subtotal: "
            f"{apples * price_per_apple} - {discount} = {total}.\n"
            f"#### {total}"
        )
        chosen_response = (
            f"<think>\n"
            f"Step 1: Cost of apples = {apples} * {price_per_apple} = {apples * price_per_apple}.\n"
            f"Step 2: Apply discount = {apples * price_per_apple} - {discount} = {total}.\n"
            f"</think>\n"
            f"<answer>\n\\boxed{{{total}}}\n</answer>"
        )
        # 构造负样本：冗长且最后一步把减法算成加法（用于训练 DPO/SimPO 消除长度偏见与计算错误）
        wrong_total = apples * price_per_apple + discount
        rejected_response = (
            f"<think>\n"
            f"Let us think very carefully about Alice and her shopping trip to buy apples. "
            f"Apples are very healthy fruits. She buys {apples} apples and each one costs ${price_per_apple}. "
            f"So multiplying {apples} by {price_per_apple} gives {apples * price_per_apple}. "
            f"Then there is a coupon of ${discount}, so we add {discount} to get {wrong_total}.\n"
            f"</think>\n"
            f"<answer>\n\\boxed{{{wrong_total}}}\n</answer>"
        )

        samples.append(
            {
                "question": question,
                "raw_answer": raw_gsm8k_answer,
                "gold_answer": str(total),
                "chosen": chosen_response,
                "rejected": rejected_response,
            }
        )
    return samples


def load_sft_dataset(
    dataset_name: str = "openai/gsm8k",
    subset: str = "main",
    split: str = "train",
    max_samples: int = -1,
    use_synthetic: bool = False,
) -> Any:
    """
    加载 Stage 1 监督微调 (SFT) 数据集.
    返回包含 ['prompt', 'completion', 'text', 'gold_answer'] 字段的数据集对象.
    """
    if use_synthetic:
        raw_list = _build_synthetic_math_samples(
            num_samples=max_samples if max_samples > 0 else 32
        )
        formatted = [
            format_sft_example(item["question"], item["raw_answer"])
            for item in raw_list
        ]
        return SimpleListDataset(formatted)

    from datasets import load_dataset

    ds = load_dataset(dataset_name, subset, split=split)
    if max_samples > 0 and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    def _map_fn(row: Dict[str, Any]) -> Dict[str, str]:
        q = row.get("question", row.get("problem", ""))
        a = row.get("answer", row.get("solution", ""))
        return format_sft_example(q, a)

    return ds.map(_map_fn, remove_columns=ds.column_names, desc="Formatting SFT dataset")


def load_preference_dataset(
    dataset_name: str = "argilla/distilabel-math-preference-dpo",
    split: str = "train",
    max_samples: int = -1,
    use_synthetic: bool = False,
) -> Any:
    """
    加载 Stage 2 偏好对齐 (DPO / SimPO) 数据集.
    返回标准三元组字段：['prompt', 'chosen', 'rejected'].
    """
    if use_synthetic:
        raw_list = _build_synthetic_math_samples(
            num_samples=max_samples if max_samples > 0 else 32
        )
        pref_list = [
            {
                "prompt": format_inference_prompt(item["question"]),
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            }
            for item in raw_list
        ]
        return SimpleListDataset(pref_list)

    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    if max_samples > 0 and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    def _map_pref(row: Dict[str, Any]) -> Dict[str, str]:
        question = row.get("instruction", row.get("prompt", row.get("question", "")))
        chosen = row.get("chosen_response", row.get("chosen", ""))
        rejected = row.get("rejected_response", row.get("rejected", ""))
        if isinstance(chosen, list):
            chosen = chosen[-1].get("content", str(chosen))
        if isinstance(rejected, list):
            rejected = rejected[-1].get("content", str(rejected))
        return {
            "prompt": format_inference_prompt(str(question)),
            "chosen": str(chosen),
            "rejected": str(rejected),
        }

    return ds.map(_map_pref, remove_columns=ds.column_names, desc="Formatting Preference dataset")


def load_grpo_dataset(
    dataset_name: str = "openai/gsm8k",
    subset: str = "main",
    split: str = "train",
    max_samples: int = 1500,
    use_synthetic: bool = False,
) -> Any:
    """
    加载 Stage 3 GRPO 强化学习数据集.
    GRPO 只需要输入问题 `prompt` 和用于规则奖励判定的 `gold_answer`，不需要现成的思维链标签！
    """
    if use_synthetic:
        raw_list = _build_synthetic_math_samples(
            num_samples=max_samples if max_samples > 0 else 24
        )
        grpo_list = [
            {
                "prompt": format_inference_prompt(item["question"]),
                "question": item["question"],
                "gold_answer": item["gold_answer"],
            }
            for item in raw_list
        ]
        return SimpleListDataset(grpo_list)

    from datasets import load_dataset

    ds = load_dataset(dataset_name, subset, split=split)
    if max_samples > 0 and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    def _map_grpo(row: Dict[str, Any]) -> Dict[str, str]:
        q = row["question"]
        gold = extract_gold_answer(row["answer"])
        return {
            "prompt": format_inference_prompt(q),
            "question": q,
            "gold_answer": gold,
        }

    return ds.map(_map_grpo, remove_columns=ds.column_names, desc="Formatting GRPO dataset")


def load_eval_dataset(
    benchmark: str = "gsm8k",
    max_samples: int = 200,
    use_synthetic: bool = False,
) -> Any:
    """
    加载评测基准数据集：支持域内 (gsm8k) 与域外 OOD (svamp).
    """
    if use_synthetic:
        raw_list = _build_synthetic_math_samples(
            num_samples=max_samples if max_samples > 0 else 20
        )
        eval_list = [
            {
                "question": item["question"],
                "prompt": format_inference_prompt(item["question"]),
                "gold_answer": item["gold_answer"],
                "benchmark": benchmark,
            }
            for item in raw_list
        ]
        return SimpleListDataset(eval_list)

    from datasets import load_dataset

    if benchmark.lower() == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        if max_samples > 0 and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        return ds.map(
            lambda r: {
                "question": r["question"],
                "prompt": format_inference_prompt(r["question"]),
                "gold_answer": extract_gold_answer(r["answer"]),
                "benchmark": "gsm8k",
            },
            remove_columns=ds.column_names,
        )
    elif benchmark.lower() == "svamp":
        ds = load_dataset("ChilleD/SVAMP", split="test")
        if max_samples > 0 and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        return ds.map(
            lambda r: {
                "question": f"{r['Body'].strip()} {r['Question'].strip()}",
                "prompt": format_inference_prompt(f"{r['Body'].strip()} {r['Question'].strip()}"),
                "gold_answer": extract_gold_answer(str(r["Answer"])),
                "benchmark": "svamp",
            },
            remove_columns=ds.column_names,
        )
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}. Choose from ['gsm8k', 'svamp'].")
