"""数据加载、Prompt 模板与答案解析模块."""
from .prompt_templates import (
    SYSTEM_PROMPT,
    format_sft_example,
    format_inference_prompt,
    extract_gold_answer,
    extract_predicted_answer,
    check_format_compliance,
)
from .dataset_loader import (
    load_sft_dataset,
    load_preference_dataset,
    load_grpo_dataset,
    load_eval_dataset,
)
