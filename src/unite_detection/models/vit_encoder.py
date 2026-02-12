from importlib.util import find_spec
from typing import Callable, Literal, cast, overload, override

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn


def _check_flash_attn() -> bool:
    return find_spec("flash_attn") is not None


HAS_FLASH_ATTN = _check_flash_attn()
flash_attn_func: Callable | None  # pyright: ignore[reportMissingTypeArgument]
if HAS_FLASH_ATTN:
    from flash_attn import (  # pyright: ignore[reportMissingImports]
        flash_attn_func,  # pyright: ignore[reportUnknownVariableType]
    )
else:
    flash_attn_func = None


class ViTEncoder(nn.Module):
    def __init__(
        self, embed_size: int = 768, num_heads: int = 12, dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size, (
            "embed_size must be divisible by num_heads"
        )

        self.q_proj: nn.Module = nn.Linear(embed_size, embed_size)
        self.k_proj: nn.Module = nn.Linear(embed_size, embed_size)
        self.v_proj: nn.Module = nn.Linear(embed_size, embed_size)
        self.out_proj: nn.Module = nn.Linear(embed_size, embed_size)
        self.dropout_p: float = dropout

        self.ln1: nn.Module = nn.LayerNorm(embed_size)
        self.mlp: nn.Module = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )
        self.ln2: nn.Module = nn.LayerNorm(embed_size)

    @overload
    def forward(
        self,
        x: Float[Tensor, "batch seq_len embed_size"],
        return_attn_output: Literal[False],
    ) -> Float[Tensor, "batch seq_len embed_size"]: ...

    @overload
    def forward(
        self,
        x: Float[Tensor, "batch seq_len embed_size"],
        return_attn_output: Literal[True],
    ) -> tuple[
        Float[Tensor, "batch seq_len embed_size"],
        Float[Tensor, "batch seq_len heads h_dim"],
    ]: ...

    @override
    def forward(self, x: Float[Tensor, "b l s"], return_attn_output: bool = False):
        # x: (batch, token/frame * frames, embed_size)
        batch_size, frame_token, embed_size = x.shape

        # Project queries, keys, values
        q = cast(Float[Tensor, "b l s"], self.q_proj(x))
        k = cast(Float[Tensor, "b l s"], self.k_proj(x))
        v = cast(Float[Tensor, "b l s"], self.v_proj(x))

        # Split into multiple heads
        # (batch, token/frame * frames, num_heads, head_dim)
        q_in: Float[Tensor, "b l h d"] = q.view(
            batch_size, frame_token, self.num_heads, self.head_dim
        )
        k_in: Float[Tensor, "b l h d"] = k.view(
            batch_size, frame_token, self.num_heads, self.head_dim
        )
        v_in: Float[Tensor, "b l h d"] = v.view(
            batch_size, frame_token, self.num_heads, self.head_dim
        )

        # Apply scaled dot product attention
        # dropout_p는 훈련 중일 때만 적용
        if HAS_FLASH_ATTN and q.is_cuda:
            assert flash_attn_func is not None
            attn_output_raw = cast(
                Float[Tensor, "b l h d"],
                flash_attn_func(
                    q_in,
                    k_in,
                    v_in,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    softmax_scale=None,  # None이면 1/sqrt(head_dim) 자동 적용
                    causal=False,
                ),
            )
        else:
            q_t: Float[Tensor, "b h l d"] = q_in.transpose(1, 2)
            k_t: Float[Tensor, "b h l d"] = k_in.transpose(1, 2)
            v_t: Float[Tensor, "b h l d"] = v_in.transpose(1, 2)

            attn_output_raw_t: Float[Tensor, "b h l d"] = (
                F.scaled_dot_product_attention(
                    q_t,
                    k_t,
                    v_t,
                    attn_mask=None,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False,
                )
            )
            attn_output_raw: Float[Tensor, "b l h d"] = attn_output_raw_t.transpose(
                1, 2
            )

        # attn_output: (batch, token/frame * frames, num_heads, head_dim)

        # Concatenate heads and apply final linear projection
        attn_output: Float[Tensor, "b l s"] = attn_output_raw.contiguous().view(
            batch_size, frame_token, embed_size
        )
        attn_output_proj = cast(Float[Tensor, "b l s"], self.out_proj(attn_output))

        x_ln1 = cast(
            Float[Tensor, "b l s"], self.ln1(x + attn_output_proj)
        )  # Residual connection + LayerNorm
        x_ln2 = cast(Float[Tensor, "b l s"], self.ln2(x_ln1 + self.mlp(x_ln1)))

        if return_attn_output:
            return x_ln2, attn_output_raw
        return x_ln2
