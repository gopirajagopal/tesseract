from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .normalization import RMSNorm
from .embeddings import TokenEmbedding
from .attention import GroupedQueryAttention
from .feedforward import SwiGLUFFN


class TransformerBlock(nn.Module):
    """Single transformer decoder block.

    Pre-norm architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = GroupedQueryAttention(config)
        self.ffn = SwiGLUFFN(config)
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self-attention (pre-norm)
        residual = hidden_states
        hidden_states, past_key_value = self.attention(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # FFN (pre-norm)
        residual = hidden_states
        hidden_states = residual + self.ffn(self.ffn_norm(hidden_states))

        return hidden_states, past_key_value


class Transformer(nn.Module):
    """Full decoder-only transformer.

    Architecture mirrors Claude / LLaMA:
    - Token embedding
    - N × TransformerBlock (GQA + SwiGLU, RMSNorm, RoPE)
    - Final RMSNorm
    - LM head (optionally tied to embedding weights)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count trainable parameters. Optionally exclude embedding tables."""
        if non_embedding:
            return sum(
                p.numel() for n, p in self.named_parameters()
                if p.requires_grad and "embed_tokens" not in n and "lm_head" not in n
            )
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens.embedding

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embed_tokens.embedding = new_embeddings

    def _make_causal_mask(
        self, seq_len: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Upper-triangular additive causal mask."""
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0).contiguous()  # (1, 1, T, T)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids:        (batch, seq_len)
            attention_mask:   (batch, seq_len) padding mask (1=keep, 0=mask); optional
            position_ids:     (batch, seq_len); auto-generated if None
            past_key_values:  list of per-layer (k, v) caches for incremental decoding
            use_cache:        return updated KV caches
            return_hidden_states: return all layer hidden states

        Returns dict with:
            logits:           (batch, seq_len, vocab_size)
            past_key_values:  updated caches (if use_cache)
            hidden_states:    list of layer outputs (if return_hidden_states)
        """
        B, T = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # Position IDs
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + T, device=device).unsqueeze(0)

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Causal + padding mask
        causal_mask = self._make_causal_mask(T, dtype, device)
        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask.float()).masked_fill(
                attention_mask == 0, float("-inf")
            )
            causal_mask = causal_mask + pad_mask[:, None, None, :]

        all_hidden_states = [] if return_hidden_states else None
        new_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None

            if self.config.use_gradient_checkpointing and self.training:
                hidden_states, kv = checkpoint(
                    layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_kv,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, kv = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )

            if use_cache and kv is not None:
                new_cache.append(kv)

            if return_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        result: Dict[str, object] = {"logits": logits}
        if use_cache:
            result["past_key_values"] = new_cache
        if return_hidden_states:
            result["hidden_states"] = all_hidden_states
        return result

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Return the total number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            total -= self.embed_tokens.embedding.weight.numel()
        return total
