"""三阶段核心后训练与强化对齐训练器模块."""
from .sft_trainer import run_sft_training
from .pref_trainer import run_preference_training, compute_dpo_or_simpo_loss
from .grpo_trainer import run_grpo_training, compute_rule_rewards
