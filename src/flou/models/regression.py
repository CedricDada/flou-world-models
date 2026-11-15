"""Famille A : régression. U-Net, perte MSE, une seule passe. Sortie déterministe.

Sous une perte quadratique, le meilleur prédicteur est l'espérance conditionnelle E[y | x] (cours,
chap. 2) : quand le futur a deux issues, c'est le mélange pA + (1-p)B, donc du flou. La variante
`oracle=True` reçoit en plus la branche réellement tirée (deux plans constants one-hot) : c'est le
contrôle H0c, qui doit prédire des images nettes (moins de 5 % de flou).

Images en [-1, 1] ; la sortie est linéaire et bornée à [-1, 1] à la prédiction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import UNetConfig
from .unet import UNet


def branch_planes(branch: torch.Tensor, size: int = 64) -> torch.Tensor:
    """(B,) de -1/0/1 -> (B, 2, size, size) : plan A (branche 1) et plan B (branche 0) ; zéros si -1."""
    planes = torch.stack([(branch == 1), (branch == 0)], dim=1).float()
    return planes[:, :, None, None].expand(-1, -1, size, size)


class RegressionModel(nn.Module):
    def __init__(self, k_ctx: int = 4, oracle: bool = False, unet: UNetConfig | None = None):
        super().__init__()
        self.k_ctx, self.oracle = k_ctx, oracle
        cfg = unet or UNetConfig(in_ch=3 * k_ctx, n_extra_cond=2 if oracle else 0)
        self.net = UNet(cfg)

    def forward(self, ctx: torch.Tensor, action: torch.Tensor, branch: torch.Tensor | None = None) -> torch.Tensor:
        extra = None
        if self.oracle:
            if branch is None:
                raise ValueError("le régresseur oracle exige la branche")
            extra = branch_planes(branch, ctx.shape[-1])
        return self.net(ctx.flatten(1, 2), action, None, extra)

    def loss(self, ctx, action, target, branch=None):
        pred = self(ctx, action, branch)
        loss = F.mse_loss(pred.float(), target.float())
        return loss, {"mse": loss.detach()}

    @torch.no_grad()
    def predict(self, ctx, action, generator=None, branch=None, **kw) -> torch.Tensor:
        return self(ctx, action, branch).float().clamp(-1.0, 1.0)
