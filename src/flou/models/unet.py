"""U-Net partagé par la famille A (régression) et la famille B (débruiteur EDM).

Construit en parcourant `config.unet_plan` : blocs résiduels avec normalisation de groupe adaptative
(AdaGN) conditionnée par l'embedding de l'action (+ celui de c_noise = ln σ / 4 pour le débruiteur).
Pas d'attention : la tâche (suivre quelques objets de quelques pixels) est locale, et chaque niveau
supplémentaire coûte du calcul de T4 (voir docs/EXPERIMENTS.md).

NON EXÉCUTÉ dans l'environnement où ce code a été écrit (pas de PyTorch) : `scripts/smoke_test.py`
construit chaque modèle, vérifie les formes, compare le nombre de paramètres au compteur `params.py`
et lance un pas d'entraînement. À lancer en premier sur la machine d'entraînement.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import UNetConfig, check_plan, unet_plan


def fourier_embedding(c: torch.Tensor, dim: int, fmin: float = 1.0, fmax: float = 64.0) -> torch.Tensor:
    """Embedding sinusoïdal d'un scalaire par exemple : (B,) -> (B, dim). Fréquences géométriques fixes."""
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(fmin), math.log(fmax), half, device=c.device, dtype=torch.float32))
    arg = 2.0 * math.pi * c.float()[:, None] * freqs[None, :]
    return torch.cat([arg.sin(), arg.cos()], dim=1)


class ResBlock(nn.Module):
    """GN -> SiLU -> conv ; AdaGN(emb) -> SiLU -> conv ; + saut. Comptage : voir params.res_block."""

    def __init__(self, cin: int, cout: int, emb_dim: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, cout, affine=False)
        self.proj = nn.Linear(emb_dim, 2 * cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.conv2.weight)          # chaque bloc démarre comme l'identité
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.proj(F.silu(emb)).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return self.skip(x) + h


class Up(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class OutHead(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int):
        super().__init__()
        self.norm = nn.GroupNorm(groups, cin)
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.silu(self.norm(x)))


class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        self.plan = unet_plan(cfg)
        check_plan(self.plan, cfg)
        e, g = cfg.emb_dim, cfg.groups
        self.act_emb = nn.Embedding(cfg.n_actions, e)
        if cfg.sigma_cond:
            self.noise_mlp = nn.Sequential(nn.Linear(e, e), nn.SiLU(), nn.Linear(e, e))
        layers = []
        for op in self.plan:
            k = op[0]
            if k == "conv_in":
                layers.append(nn.Conv2d(op[1], op[2], 3, padding=1))
            elif k in ("res_enc", "res_mid"):
                layers.append(ResBlock(op[1], op[2], e, g))
            elif k == "res_dec":
                layers.append(ResBlock(op[1] + op[3], op[2], e, g))
            elif k == "down":
                layers.append(nn.Conv2d(op[1], op[1], 3, stride=2, padding=1))
            elif k == "up":
                layers.append(Up(op[1]))
            elif k == "conv_out":
                layers.append(OutHead(op[1], op[2], g))
            else:
                raise ValueError(op)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, action: torch.Tensor, c_noise: torch.Tensor | None = None,
                extra: torch.Tensor | None = None) -> torch.Tensor:
        """x : (B, in_ch, 64, 64) ; action : (B,) long ; c_noise : (B,) (débruiteur) ;
        extra : (B, n_extra_cond, 64, 64) canaux de conditionnement constants (oracle)."""
        emb = self.act_emb(action)
        if self.cfg.sigma_cond:
            if c_noise is None:
                raise ValueError("c_noise requis pour un débruiteur")
            emb = emb + self.noise_mlp(fourier_embedding(c_noise.reshape(-1), self.cfg.emb_dim).to(emb.dtype))
        h = x if extra is None else torch.cat([x, extra.to(x.dtype)], dim=1)
        skips = []
        for op, layer in zip(self.plan, self.layers):
            k = op[0]
            if k == "conv_in" or k == "down":
                h = layer(h)
                skips.append(h)
            elif k == "res_enc":
                h = layer(h, emb)
                skips.append(h)
            elif k == "res_mid":
                h = layer(h, emb)
            elif k == "res_dec":
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
            else:                                    # up, conv_out
                h = layer(h)
        return h
