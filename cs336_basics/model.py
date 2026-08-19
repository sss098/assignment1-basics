import math

import torch
from torch import nn


class Linear(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            device=None,
            dtype=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty(
            out_features,
            in_features,
            device=device,
            dtype=dtype,
        )

        self.weight = nn.Parameter(weight)

        std = math.sqrt(
            2.0 / (in_features + out_features)
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:

        return x @ self.weight.T


class Embedding(nn.Module):

    def __init__(
            self,
            num_embeddings:int,
            embedding_dim:int,
            device=None,
            dtype=None,
    ):
        super().__init__()

        weight = torch.empty(
            num_embeddings,
            embedding_dim,
            device=device,
            dtype=dtype,
        )

        self.weight = nn.Parameter(weight)

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(
            self,
            token_ids: torch.Tensor,
    ) -> torch.Tensor:

        return self.weight[token_ids]


class RMSNorm(nn.Module):

    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device=None,
            dtype=None,
    ):
            super().__init__()

            self.d_model = d_model
            self.eps = eps

            self.weight = nn.Parameter(
                 torch.ones(
                      d_model,
                      device=device,
                      dtype=dtype,
                 )
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        input_type = x.dtype
        x_float = x.to(torch.float32)
        mean_square = torch.mean(
            x_float ** 2,
            dim=-1,
            keepdim=True,
        )

        rms = torch.sqrt(
            mean_square + self.eps
        )

        normalized = x_float / rms
        normalized = normalized.to(input_type)

        return normalized * self.weight

def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU 激活函数。

    公式:
    SiLU(x) = x * sigmoid(x)

    返回:
        [b"the"]
    """
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):

    def __init__(
            self,
            d_model,
            d_ff,
            device=None,
            dtype=None,
    ):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.w2(
            silu(self.w1(x)) * self.w3(x)
        )

class RotaryPositionalEmbedding(nn.Module):

    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device:None,
    ):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        dim_indices = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )

        inv_freq = theta ** (
            -dim_indices / d_k
        )

        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        angles = (
            positions[:, None] * inv_freq[None, :]
        )

        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)

        self.register_buffer(
            "cos_cache",
            cos_cache,
            persistent=False,
        )

        self.register_buffer(
            "sin_cache",
            sin_cache,
            persistent=False,
        )

    def forward(
            self,
            x: torch.tensor,
            token_positions: torch.tensor,
    ) -> torch.Tensor:

        token_positions = token_positions.to(
            device=self.cos_cache.device,
        )

        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        while cos.ndim < x_even.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        rotated_even = (
            x_even * cos
            - x_odd *sin
        )

        rotate_odd=(
            x_even * sin
            + x_odd *cos
        )

        output = torch.stack(
            [rotated_even, rotate_odd],
            dim=-1
        )

        return output.flatten(-2)

def softmax(
        x: torch.Tensor,
        dim: int,
) -> torch.Tensor:

    max_value = torch.max(
        x,
        dim=dim,
        keepdim=True,
    ).values

    shifted_x = x - max_value

    exp_x = torch.exp(shifted_x)

    sum_exp = torch.sum(
        exp_x,
        dim=dim,
        keepdim=True,
    )

    return exp_x / sum_exp

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
) -> torch.Tensor:

    d_k = Q.shape[-1]

    scores = (
        Q @ K.transpose(-2, -1)
    ) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

    attention_weights = softmax(
        scores,
        dim=-1,
    )

    output = attention_weights @ V

    return output

class CausalMultiHeadSelfAttention(nn.Module):

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int |None = None,
            theta: float | None = None,
            device=None,
            dtype=None,
    ):

        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.rope = None

        if(
            max_seq_len is not None
            and theta is not None
        ):
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
                device=device,
            )

    def _split_heads(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        x = x.reshape(
            *x.shape[:-2],
            seq_len,
            self.num_heads,
            self.d_head,
        )

        x = x.transpose(-3,-2,)

        return x

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        seq_len = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        if self.rope is not None:

            if token_positions is None:
                token_positions = torch.arange(
                    seq_len,
                    device=x.device,
                )

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )
        )

        attention_output = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask=causal_mask,
        )

        attention_output = (
            attention_output.transpose(-3, -2)
        )

        attention_output = attention_output.reshape(
            *x.shape[:-2],
            seq_len,
            self.d_model,
        )

        output = self.output_proj(attention_output)

        return output

class TransformerBlock(nn.Module):

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float,
            device=None,
            dtype=None,
    ):
        super().__init__()

        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )

        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )


    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        residual = x

        x = self.ln1(x)

        x = self.attn(x, token_positions=token_positions)

        x = x + residual

        residual = x

        x = self.ln2(x)

        x = self.ffn(x)

        x = x + residual

        return x

class TransformerLM(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device=None,
            dtype=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
        )
        for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
            self,
            in_indices: torch.Tensor,
    ) -> torch.Tensor:

        seq_len = in_indices.shape[-1]

        if seq_len > self.context_length:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds context length {self.context_length}"
            )

        x = self.token_embeddings(in_indices)

        token_positions= torch.arange(
            seq_len,
            device=x.device,
        )

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits

def cross_entropy(
        inputs: torch.Tensor,
        targets: torch.Tensor,
) -> torch.Tensor:

    max_value = torch.max(
        inputs,
        dim=-1,
        keepdim=True,
    ).values

    shifted_inputs = inputs - max_value

    log_sum_exp = torch.log(
        torch.sum(
            torch.exp(shifted_inputs),
            dim=-1,
        )
    )

    target_logits = torch.gather(
        shifted_inputs,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = log_sum_exp - target_logits

    return losses.mean()