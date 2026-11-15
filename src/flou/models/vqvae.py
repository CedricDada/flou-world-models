"""VQ-VAE (van den Oord et al. 2017) : quantification, trois termes de perte, straight-through.

    L = ||x - D(z_q)||²  +  ||sg[z_e] - e||²  +  β ||z_e - sg[e]||²        (β = 0,25)

La perplexité de l'usage du codebook, exp(-Σ π_k ln π_k), vaut K pour un usage uniforme et 1 si un seul
code est utilisé : c'est le diagnostic d'effondrement du codebook (cours, chap. 4). Les codes inutilisés
peuvent être réinitialisés sur des sorties récentes du codeur (`restart_dead_codes`) : pratique courante,
non issue des sources lues pour le cours ; elle est activée par défaut et doit être rapportée.

La quantification est calculée en float32 même sous précision mixte (distances sensibles).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import VQConfig, vq_plan


class VQBlock(nn.Module):
    def __init__(self, c: int, groups: int):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(groups, c), nn.Conv2d(c, c, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(groups, c), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        return x + self.c2(F.silu(self.n2(h)))


class _Up(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class _DecOut(nn.Module):
    def __init__(self, c: int, cout: int, groups: int):
        super().__init__()
        self.norm, self.conv = nn.GroupNorm(groups, c), nn.Conv2d(c, cout, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.silu(self.norm(x)))


def perplexity(idx: torch.Tensor, k: int) -> torch.Tensor:
    pi = torch.bincount(idx.reshape(-1), minlength=k).float()
    pi = pi / pi.sum()
    nz = pi[pi > 0]
    return torch.exp(-(nz * nz.log()).sum())


class VQVAE(nn.Module):
    def __init__(self, cfg: VQConfig):
        super().__init__()
        self.cfg = cfg
        g = cfg.groups
        self.codebook = nn.Embedding(cfg.K, cfg.dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / cfg.K, 1.0 / cfg.K)
        enc, dec = [], []
        for op in vq_plan(cfg):
            k = op[0]
            if k == "enc_conv_in":
                enc.append(nn.Conv2d(op[1], op[2], 3, padding=1))
            elif k == "enc_block":
                enc.append(VQBlock(op[1], g))
            elif k == "enc_down":
                enc.append(nn.Conv2d(op[1], op[1], 3, stride=2, padding=1))
            elif k == "enc_proj":
                enc.append(nn.Conv2d(op[1], op[2], 1))
            elif k == "dec_proj":
                dec.append(nn.Conv2d(op[1], op[2], 1))
            elif k == "dec_block":
                dec.append(VQBlock(op[1], g))
            elif k == "dec_up":
                dec.append(_Up(op[1]))
            elif k == "dec_conv_out":
                dec.append(_DecOut(op[1], op[2], g))
            else:
                raise ValueError(op)
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)
        self.register_buffer("usage", torch.full((cfg.K,), 1.0 / cfg.K))   # fréquence d'usage (moyenne mobile)

    # ---- quantification ----------------------------------------------------------------
    def quantize(self, z_e: torch.Tensor):
        """z_e : (B, dim, g, g) -> (z_q straight-through, indices (B, g*g), perte VQ, z_e aplati)."""
        b, d, h, w = z_e.shape
        z = z_e.float().permute(0, 2, 3, 1).reshape(b, h * w, d)
        with torch.autocast(device_type=z.device.type, enabled=False):
            dist = torch.cdist(z, self.codebook.weight.float()[None].expand(b, -1, -1))   # (B, N, K)
            idx = dist.argmin(dim=-1)
            e = self.codebook(idx)                                                        # (B, N, dim)
            loss = F.mse_loss(e, z.detach()) + self.cfg.beta * F.mse_loss(z, e.detach())
            zq = z + (e - z).detach()                                                     # straight-through
        zq = zq.reshape(b, h, w, d).permute(0, 3, 1, 2)
        return zq, idx, loss, z

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)
        zq, idx, vq_loss, z = self.quantize(z_e)
        xhat = self.decoder(zq.to(z_e.dtype))
        recon = F.mse_loss(xhat.float(), x.float())
        return xhat, idx, recon, vq_loss, z

    def loss(self, x: torch.Tensor):
        _, idx, recon, vq_loss, z = self.forward(x)
        with torch.no_grad():
            counts = torch.bincount(idx.reshape(-1), minlength=self.cfg.K).float()
            self.usage.mul_(0.99).add_(0.01 * counts / counts.sum())
            ppl = perplexity(idx, self.cfg.K)
        return recon + vq_loss, {"recon": recon.detach(), "vq": vq_loss.detach(), "perplexity": ppl,
                                 "_z": z.detach()}

    @torch.no_grad()
    def restart_dead_codes(self, z: torch.Tensor, threshold: float = 0.03) -> int:
        """Remplace les codes d'usage < threshold / K par des sorties du codeur tirées au hasard (+ bruit)."""
        dead = (self.usage < threshold / self.cfg.K).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            return 0
        flat = z.reshape(-1, z.shape[-1])
        pick = torch.randint(0, flat.shape[0], (len(dead),), device=flat.device)
        self.codebook.weight[dead] = flat[pick] + 0.01 * torch.randn_like(flat[pick])
        self.usage[dead] = 1.0 / self.cfg.K
        return int(len(dead))

    # ---- jetons <-> images ---------------------------------------------------------------
    @torch.no_grad()
    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, 3, 64, 64) dans [-1, 1] -> (N, g*g) jetons (long)."""
        _, idx, _, _ = self.quantize(self.encoder(x))
        return idx

    @torch.no_grad()
    def decode_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        b, n = idx.shape
        g = self.cfg.grid
        e = self.codebook(idx).reshape(b, g, g, self.cfg.dim).permute(0, 3, 1, 2)
        return self.decoder(e).float().clamp(-1.0, 1.0)
