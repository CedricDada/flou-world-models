"""Sonde de position apprise (CNN) : régresse la position (x, y) de l'agent, de l'obstacle, de l'ennemi
et du petit objet à partir d'UNE image. Entraînée sur de vraies images dont l'état exact est connu, puis
validée avant de classer quoi que ce soit : RMSE sous 0,5 pixel sur des images de validation (cours,
chap. 7 : « un R² de 0,99 sur 64 pixels ≈ 1,85 pixel d'erreur »). Sa référence de plancher est la sonde
analytique (`flou.eval.probe.AnalyticProbe`).

Sortie en pixels, dans le repère de `flou.eval.probe.true_positions` (centres de pixels).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class PositionProbe(nn.Module):
    def __init__(self, width: int = 32, groups: int = 8):
        super().__init__()
        w = width

        def block(cin, cout, stride):
            return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.GroupNorm(groups, cout), nn.SiLU()]

        layers = block(3, w, 1) + block(w, 2 * w, 2) + block(2 * w, 2 * w, 1) + block(2 * w, 4 * w, 2) \
            + block(4 * w, 4 * w, 1) + block(4 * w, 4 * w, 2)                      # 64 -> 8
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * w * 8 * 8, 256), nn.SiLU(), nn.Linear(256, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 3, 64, 64) dans [-1, 1] -> (B, 4, 2) positions en pixels."""
        return (self.head(self.features(x)) * 64.0).reshape(-1, 4, 2)

    def loss(self, x: torch.Tensor, pos: torch.Tensor):
        pred = self(x)
        loss = F.mse_loss(pred.float(), pos.float()) / 64.0 ** 2 * 100.0      # échelle ~ 1 en début d'entraînement
        rmse = ((pred.float() - pos.float()) ** 2).sum(-1).mean().sqrt()
        return loss, {"loss": loss.detach(), "rmse_px": rmse.detach()}


class ProbeAdapter:
    """Interface de `AnalyticProbe` : imgs uint8 (N, 64, 64, 3) -> positions (N, 4, 2) NumPy."""

    def __init__(self, probe: PositionProbe, device: str = "cuda", batch: int = 512):
        self.probe, self.device, self.batch = probe.to(device).eval(), device, batch

    @torch.no_grad()
    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        imgs = np.asarray(imgs)
        if imgs.dtype != np.uint8:
            imgs = np.clip(np.round(imgs * 255.0), 0, 255).astype(np.uint8)
        out = []
        for i in range(0, len(imgs), self.batch):
            x = torch.from_numpy(imgs[i:i + self.batch]).to(self.device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            out.append(self.probe(x).float().cpu().numpy())
        return np.concatenate(out)
