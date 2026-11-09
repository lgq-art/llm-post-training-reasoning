"""
标准化推理 Prompt 模板与数学答案解析工具.

科研设计说明：
为了统一评测 SFT、DPO/SimPO 与 GRPO (DeepSeek-R1 范式)，我们强制模型采用结构化思考标签：
<think>
分步逻辑推导过程...
</think>
<answer>
\\boxed{最终数值}
</answer>
这种设计不仅便于自动化正则提取答案，还能精确统计“思维链 Token 长度 (Reasoning Tokens)”与“格式遵循率”。
"""

import re
from typing import Optional, Dict, Any

SYSTEM_PROMPT = (
    "You are a rigorous mathematical reasoning assistant. "
    "Before giving the final answer, always think step by step inside <think> and </think> tags. "
    "Then provide the final numerical answer inside <answer> and </answer> tags, enclosed in \\boxed{}."
)


def extract_gold_answer(raw_answer_str: str) -> str:
    """
    从 GSM8K 原始答案字符串（如 '... #### 42'）或纯数字中提取标准数值答案.
    """
    if "####" in raw_answer_str:
        gold = raw_answer_str.split("####")[-1].strip()
    else:
        gold = str(raw_answer_str).strip()
    # 去除千分位逗号和多余空格（如 "1,250" -> "1250"）
    gold = gold.replace(",", "").replace("$", "").strip()
    return gold


def extract_cot_and_gold(raw_answer_str: str) -> tuple[str, str]:
    """
    将 GSM8K 的 answer 字段拆分为 (思维链推导过程, 最终标准答案).
    """
    if "####" in raw_answer_str:
        parts = raw_answer_str.split("####")
        cot = parts[0].strip()
        gold = parts[-1].strip().replace(",", "")
    else:
        cot = raw_answer_str.strip()
        gold = extract_predicted_answer(raw_answer_str) or ""
    return cot, gold


def format_sft_example(question: str, raw_answer: str) -> Dict[str, str]:
    """
    将单条 (问题, 原始解答) 格式化为 SFT 监督微调所需的 Prompt + Completion 结构.
    """
    cot, gold = extract_cot_and_gold(raw_answer)
    prompt = format_inference_prompt(question)
    completion = (
        f"<think>\n{cot}\n</think>\n"
        f"<answer>\n\\boxed{{{gold}}}\n</answer>"
    )
    return {
        "prompt": prompt,
        "completion": completion,
        "text": f"{prompt}{completion}",
        "gold_answer": gold,
    }


def format_inference_prompt(question: str) -> str:
    """
    构造测试推理或 GRPO 采样时的输入 Prompt.
    """
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nQuestion: {question.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_predicted_answer(model_output: str) -> Optional[str]:
    """
    从模型生成文本中提取预测的数值答案.
    优先级：
    1. \\boxed{...} 中的内容
    2. <answer>...</answer> 标签中的数字
    3. 文本中最后出现的数值（兜底策略，用于评测未经过格式微调的 Zero-shot 基座模型）
    """
    if not model_output:
        return None

    # 1. 匹配 \boxed{...}
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", model_output)
    if boxed_matches:
        candidate = boxed_matches[-1].strip().replace(",", "").replace("$", "")
        num_match = re.findall(r"-?\d+\.?\d*", candidate)
        return num_match[-1] if num_match else candidate

    # 2. 匹配 <answer>...</answer>
    ans_matches = re.findall(r"<answer>(.*?)</answer>", model_output, flags=re.DOTALL)
    if ans_matches:
        candidate = ans_matches[-1].strip().replace(",", "").replace("$", "")
        num_match = re.findall(r"-?\d+\.?\d*", candidate)
        return num_match[-1] if num_match else candidate

    # 3. 匹配 #### 后的数字
    if "####" in model_output:
        after_hash = model_output.split("####")[-1].strip().replace(",", "")
        num_match = re.findall(r"-?\d+\.?\d*", after_hash)
        if num_match:
            return num_match[0]

    # 4. 兜底：提取全文最后一个数字
    all_numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", model_output)
    if all_numbers:
        return all_numbers[-1].replace(",", "")

    return None


def is_answer_correct(pred: Optional[str], gold: str, tol: float = 1e-4) -> bool:
    """
    判断预测答案与真实答案是否数值等价（兼容整数与浮点数比较，如 '42.0' == '42'）.
    """
    if pred is None or gold is None:
        return False
    pred_clean = pred.strip().replace(",", "")
    gold_clean = str(gold).strip().replace(",", "")
    if pred_clean == gold_clean:
        return True
    try:
        return abs(float(pred_clean) - float(gold_clean)) <= tol
    except ValueError:
        return False


def check_format_compliance(model_output: str) -> Dict[str, Any]:
    """
    检查模型输出是否严格符合 <think>...</think><answer>\\boxed{...}</answer> 格式規範，
    并统计思考部分字符数/词数，用于 RQ2 & RQ3 的机理分析.
    """
    has_think = bool(re.search(r"<think>.*?</think>", model_output, flags=re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", model_output, flags=re.DOTALL))
    has_boxed = bool(re.search(r"\\boxed\{[^{}]+\}", model_output))
    strict_match = bool(
        re.search(
            r"^\s*<think>.*?</think>\s*<answer>.*?\\boxed\{[^{}]+\}.*?</answer>\s*$",
            model_output,
            flags=re.DOTALL,
        )
    )

    think_match = re.search(r"<think>(.*?)</think>", model_output, flags=re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else model_output
    reasoning_words = len(think_content.split())

    return {
        "has_think": has_think,
        "has_answer": has_answer,
        "has_boxed": has_boxed,
        "strict_format": strict_match,
        "reasoning_word_count": reasoning_words,
    }
