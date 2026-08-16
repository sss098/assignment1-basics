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

        weights = torch.empty(
            out_features,
            in_features,
            device=device,
            dtype=dtype,
        )

        self.weights = nn.Parameter(weights)

        std = math.sqrt(
            2.0 / (in_features + out_features)
        )

        nn.init.trunc_normal_(
            self.weights,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:

        return x @ self.weights.T


class Embedding(nn.Module):

    def __init__(
            self,
            num_embeddings:int,
            embedding_dim:int,
            device=None,
            dtype=None,
    ):
        super().__init__()

        weights = torch.empty(
            num_embeddings,
            embedding_dim,
            device=device,
            dtype=dtype,
        )

        self.weights = nn.Parameter(weights)

        nn.init.trunc_normal_(
            self.weights,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(
            self,
            token_ids: torch.Tensor,
    ) -> torch.Tensor:

        return self.weights[token_ids]


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

        nomalized = x_float / rms
        nomalized = nomalized.to(input_type)

        return nomalized * self.weight
