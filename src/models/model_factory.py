"""
模型工厂：负责加载基座大模型、配置 4-bit QLoRA 量化、注入 LoRA/DoRA 适配器，
并提供从零手写的原生 PyTorch LoRA / DoRA 线性层与微型 Transformer 构建器，
既帮助科研初学者透彻理解 LoRA 与 DoRA 的底层矩阵运算，又支持本机无依赖秒级跑通全链路单元测试。
"""

import os
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_MODULES_MAP = {
    # 仅在多头自注意力层注入低秩矩阵
    "attn_only": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # 在注意力层 + FFN (SwiGLU MLP) 全部线性层注入低秩矩阵
    "all_linear": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


@dataclass
class PeftConfigSpec:
    """通用 LoRA / DoRA 配置结构，兼容 HuggingFace PEFT 与原生 PyTorch 教学实现."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_dora: bool = False
    target_modules: List[str] = None


def build_peft_config(
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_dora: bool = False,
    target_scope: str = "all_linear",
) -> PeftConfigSpec:
    """
    构建 PEFT LoRA / DoRA 配置（对应 RQ1 核心消融变量）.
    """
    target_modules: List[str] = TARGET_MODULES_MAP.get(
        target_scope, TARGET_MODULES_MAP["all_linear"]
    )
    return PeftConfigSpec(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        target_modules=list(target_modules),
    )


def get_trainable_param_stats(model: nn.Module) -> Dict[str, Any]:
    """
    统计模型可训练参数量与总参数量占比（用于实验报告表格记录）.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    ratio = (100.0 * trainable_params / all_params) if all_params > 0 else 0.0
    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": round(ratio, 4),
    }


# ============================================================================
# 科研教学白盒实现：从零手写 LoRA 与 DoRA (Weight-Decomposed Low-Rank Adaptation)
# ============================================================================
class NativeLoRAOrDoRALinear(nn.Module):
    """
    教学级白盒实现：将普通 `nn.Linear` 替换为支持 LoRA 与 DoRA 的低秩适配层.

    数学公式对照：
    1. **LoRA (Hu et al., ICLR 2022)**:
       冻结原权重 $W_0 \\in \\mathbb{R}^{d_{\\text{out}} \\times d_{\\text{in}}}$，仅训练 $A \\in \\mathbb{R}^{r \\times d_{\\text{in}}}$ 和 $B \\in \\mathbb{R}^{d_{\\text{out}} \\times r}$：
       $$h = W_0 x + \\frac{\\alpha}{r} B A x$$
    2. **DoRA (Liu et al., ICML 2024 Oral)**:
       将权重分解为模长向量 $m \\in \\mathbb{R}^{d_{\\text{out}} \\times 1}$ 与方向矩阵 $V = W_0 + \\frac{\\alpha}{r} B A$：
       $$W_{\\text{DoRA}} = m \\cdot \\frac{W_0 + \\frac{\\alpha}{r} B A}{\\|W_0 + \\frac{\\alpha}{r} B A\\|_c}$$
       其中 $m$ 初始化为 $\\|W_0\\|_c$ 且作为可训练参数更新！
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        use_dora: bool = False,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.scaling = lora_alpha / max(r, 1)
        self.use_dora = use_dora
        self.adapter_enabled = True

        # 冻结预训练基座权重 W_0
        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.dropout = nn.Dropout(p=lora_dropout)
        # LoRA 低秩矩阵 A (Kaiming 初始化) 与 B (全零初始化，保证初始状态 ΔW = 0)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # DoRA 专属：可学习的输出通道模长向量 m (shape: [out_features, 1])
        if self.use_dora:
            with torch.no_grad():
                init_mag = self.weight.norm(p=2, dim=1, keepdim=True)
            self.dora_magnitude = nn.Parameter(init_mag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.adapter_enabled:
            return F.linear(x, self.weight, self.bias)

        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        if self.use_dora:
            new_v = self.weight + delta_w
            norm_v = new_v.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
            eff_weight = self.dora_magnitude * (new_v / norm_v)
            return F.linear(x, eff_weight, self.bias)
        else:
            base_out = F.linear(x, self.weight, self.bias)
            lora_out = F.linear(self.dropout(x), delta_w)
            return base_out + lora_out


class TinyQwen2Block(nn.Module):
    """具有标准 Qwen2 / Llama 命名规范 (q/k/v/o_proj + gate/up/down_proj) 的微型 Transformer 层."""

    def __init__(self, hidden_size: int = 64, intermediate_size: int = 128):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        B, T, C = q.shape
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(C)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        x = x + self.o_proj(attn_probs @ v)

        h2 = self.ln2(x)
        mlp_out = self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2))
        return x + mlp_out


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


class TinyQwen2CausalLM(nn.Module):
    """
    轻量级自回归语言模型（用于本机 Smoke Test，完整支持 forward、generate、disable_adapter 与 save_pretrained）.
    """

    def __init__(self, vocab_size: int = 512, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [TinyQwen2Block(hidden_size, hidden_size * 2) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def inject_peft(self, peft_cfg: PeftConfigSpec) -> None:
        # 冻结所有原始参数
        for p in self.parameters():
            p.requires_grad = False

        # 在目标模块注入 NativeLoRAOrDoRALinear
        for layer in self.layers:
            for mod_name in peft_cfg.target_modules:
                if hasattr(layer, mod_name):
                    old_linear = getattr(layer, mod_name)
                    new_linear = NativeLoRAOrDoRALinear(
                        base_linear=old_linear,
                        r=peft_cfg.r,
                        lora_alpha=peft_cfg.lora_alpha,
                        lora_dropout=peft_cfg.lora_dropout,
                        use_dora=peft_cfg.use_dora,
                    )
                    setattr(layer, mod_name, new_linear)

    @contextmanager
    def disable_adapter(self):
        """上下文管理器：临时关闭 LoRA/DoRA 适配器，退化为冻结参考模型 π_ref（用于 DPO 与 GRPO KL 计算）."""
        adapters = [m for m in self.modules() if isinstance(m, NativeLoRAOrDoRALinear)]
        for m in adapters:
            m.adapter_enabled = False
        try:
            yield
        finally:
            for m in adapters:
                m.adapter_enabled = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.95,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        seq = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.forward(seq)
            next_logits = out.logits[:, -1, :]
            if do_sample and temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            seq = torch.cat([seq, next_token], dim=1)
        return seq

    def save_pretrained(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        adapter_state = {
            k: v.cpu() for k, v in self.state_dict().items() if "lora_" in k or "dora_" in k
        }
        torch.save(adapter_state, os.path.join(save_dir, "adapter_model.pt"))


class SimpleAsciiTokenizer:
    """
    轻量级离线词级分词器（仅在 --mode smoke_test 本机离线联调时使用）.
    按单词与标签切分 Token，确保序列长度紧凑且能真实学习问答映射。
    """

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.pad_token = "<|pad|>"
        self.eos_token = "<|im_end|>"
        self.bos_token = "<|im_start|>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.padding_side = "right"
        self._id_to_tok: Dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        import re

        tokens = re.findall(r"<[^>]+>|\\boxed|\d+|\w+|[^\w\s]", str(text))
        ids = []
        for tok in tokens:
            tid = (sum(ord(c) * (i + 1) for i, c in enumerate(tok)) % (self.vocab_size - 10)) + 10
            self._id_to_tok[tid] = tok
            ids.append(tid)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        words = []
        for tid in token_ids:
            if tid in (self.pad_token_id, self.eos_token_id, self.bos_token_id):
                continue
            words.append(self._id_to_tok.get(tid, str(tid)))
        return " ".join(words)

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]


def load_model_and_tokenizer(
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B",
    use_4bit: bool = True,
    peft_config: Optional[PeftConfigSpec] = None,
    adapter_path: Optional[str] = None,
    use_tiny_mock: bool = False,
) -> Tuple[nn.Module, Any]:
    """
    加载模型与分词器，并根据配置挂载 QLoRA / DoRA 适配器.
    """
    if use_tiny_mock:
        model = TinyQwen2CausalLM(vocab_size=512, hidden_size=64, num_layers=2)
        tokenizer = SimpleAsciiTokenizer(vocab_size=512)
        if peft_config is not None:
            model.inject_peft(peft_config)
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    compute_dtype = (
        torch.bfloat16
        if (has_cuda and torch.cuda.is_bf16_supported())
        else (torch.float16 if has_cuda else torch.float32)
    )

    quantization_config = None
    if use_4bit and has_cuda:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto" if has_cuda else None,
        trust_remote_code=True,
    )

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    elif peft_config is not None:
        hf_lora_cfg = LoraConfig(
            r=peft_config.r,
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=peft_config.use_dora,
            target_modules=peft_config.target_modules,
        )
        model = get_peft_model(model, hf_lora_cfg)

    return model, tokenizer
