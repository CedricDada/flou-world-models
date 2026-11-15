"""Famille B : débruiteur EDM (Karras et al. 2022) conditionné par les images passées et l'action.

    D(x; σ, c) = c_skip(σ) x + c_out(σ) F( c_in(σ) [x ; contexte], ln σ / 4, action )

Entraînement : σ = exp(-1,2 + 1,2 ξ), perte pondérée λ(σ) = 1/c_out². Un SEUL débruiteur par graine et
par p ; l'échantillonneur d'Euler est ensuite appliqué avec n ∈ {1, 3, 5} et la référence n = 20, ce qui
rend les comparaisons entre n appariées (cours, chap. 9). Avec n = 1, l'unique pas rend D(x_σmax ; σmax),
c'est-à-dire la moyenne conditionnelle : du flou, comme la régression.

Les formules sont celles de `edm_math` (testées en NumPy) ; seul l'enrobage PyTorch est nouveau.
"""
from __future__ import annotations

import torch
from torch import nn

from . import edm_math as E
from .config import UNetConfig
from .unet import UNet


class EDMDenoiser(nn.Module):
    def __init__(self, k_ctx: int = 4, sigma_data: float = E.SIGMA_DATA, unet: UNetConfig | None = None):
        super().__init__()
        self.k_ctx, self.sigma_data = k_ctx, sigma_data
        cfg = unet or UNetConfig(in_ch=3 * (k_ctx + 1), sigma_cond=True)
        self.net = UNet(cfg)

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, ctx: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """x : (B,3,H,W) bruité ; sigma : (B,1,1,1) float32 ; ctx : (B,k,3,H,W) dans [-1,1]."""
        c_skip, c_out, c_in = E.precond_coefficients(sigma.float(), self.sigma_data)
        inp = torch.cat([c_in * x, ctx.flatten(1, 2)], dim=1)
        f = self.net(inp, action, sigma.float().reshape(-1).log() / 4.0)
        return c_skip * x + c_out * f.float()

    def loss(self, ctx, action, target, branch=None):
        b = target.shape[0]
        sigma = (E.P_MEAN + E.P_STD * torch.randn(b, 1, 1, 1, device=target.device)).exp()
        noisy = target + sigma * torch.randn_like(target)
        d = self.denoise(noisy, sigma, ctx, action)
        w = E.loss_weight(sigma, self.sigma_data)
        loss = (w * (d.float() - target.float()) ** 2).mean()
        return loss, {"edm": loss.detach()}

    @torch.no_grad()
    def sample(self, ctx, action, n_steps: int = 3, generator: torch.Generator | None = None,
               sigma_max: float = E.SIGMA_MAX, sigma_min: float = E.SIGMA_MIN, rho: float = E.RHO) -> torch.Tensor:
        """Échantillonneur d'Euler : x <- x + (σ' - σ)(x - D(x; σ))/σ, de σmax jusqu'à 0."""
        sig = E.karras_sigmas(n_steps, sigma_max, sigma_min, rho)
        b, (h, w) = ctx.shape[0], ctx.shape[-2:]
        x = torch.randn(b, 3, h, w, device=ctx.device, generator=generator) * float(sig[0])
        for s, s_next in zip(sig[:-1], sig[1:]):
            s_t = torch.full((b, 1, 1, 1), float(s), device=ctx.device, dtype=torch.float32)
            d = self.denoise(x, s_t, ctx, action).float()
            x = x + (float(s_next) - float(s)) * (x - d) / float(s)
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def predict(self, ctx, action, generator=None, branch=None, n_steps: int = 3, **kw) -> torch.Tensor:
        return self.sample(ctx, action, n_steps, generator)
