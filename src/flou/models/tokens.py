"""Famille C : VQ-VAE puis Transformer causal sur la grille de jetons. Deux modèles à entraîner.

Le VQ-VAE est entraîné d'abord (stade « vq »), puis gelé ; toutes les images de la banque sont alors
converties en jetons une fois pour toutes, et le Transformer est entraîné sur ces jetons (stade « gpt »).
"""
from __future__ import annotations

import torch
from torch import nn

from .config import GPTConfig, VQConfig
from .gpt import GPT
from .vqvae import VQVAE


class TokenWorldModel(nn.Module):
    def __init__(self, vq: VQConfig, k_ctx: int = 4, gpt: GPTConfig | None = None):
        super().__init__()
        self.k_ctx = k_ctx
        self.vq = VQVAE(vq)
        self.gpt = GPT(gpt or GPTConfig(K=vq.K, n_tokens=vq.n_tokens, ctx_frames=k_ctx))
        if self.gpt.cfg.K != vq.K or self.gpt.cfg.n_tokens != vq.n_tokens or self.gpt.cfg.ctx_frames != k_ctx:
            raise ValueError("GPTConfig incompatible avec VQConfig / k_ctx")

    @torch.no_grad()
    def tokenize_images(self, frames_nhwc_uint8: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        """(N, H, W, 3) uint8 -> (N, G) jetons (int16 pour économiser la mémoire GPU)."""
        was = self.vq.training
        self.vq.eval()
        out = []
        for i in range(0, frames_nhwc_uint8.shape[0], chunk):
            x = frames_nhwc_uint8[i:i + chunk].permute(0, 3, 1, 2).float() / 127.5 - 1.0
            out.append(self.vq.tokens(x).to(torch.int16))
        self.vq.train(was)
        return torch.cat(out)

    @torch.no_grad()
    def predict(self, ctx, action, generator=None, branch=None, tau: float = 1.0, **kw) -> torch.Tensor:
        """ctx : (B, k, 3, H, W) dans [-1, 1] -> image suivante (B, 3, H, W) dans [-1, 1]."""
        b, k = ctx.shape[:2]
        tok = self.vq.tokens(ctx.flatten(0, 1)).reshape(b, k, -1)
        nxt = self.gpt.sample(tok, action, tau, generator)
        return self.vq.decode_tokens(nxt)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction par le seul VQ-VAE (sans Transformer) : borne supérieure du rappel de la famille C."""
        return self.vq.decode_tokens(self.vq.tokens(x))
