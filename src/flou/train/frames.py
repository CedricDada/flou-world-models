"""Banque d'images sur le GPU : indexation de fenêtres (contexte, action, cible) sans passer par le CPU.

On rend une fois pour toutes toutes les images de tous les épisodes (états s_0 ... s_T) en uint8 : pour
100 000 transitions, environ 1,2 Go. Un lot est ensuite une simple indexation. L'échantillonnage est SANS
ÉTAT (graine = f(graine du run, numéro de pas)) : reprendre un entraînement au pas n redonne exactement
les mêmes lots que s'il n'avait jamais été interrompu.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.dataset import TransitionData
from ..eval.probe import true_positions
from ..game import State, render


class FrameBank:
    def __init__(self, paths, k_ctx: int, device: str = "cuda", with_positions: bool = False):
        d = TransitionData(list(paths), k=k_ctx)
        self.cfg, self.k, self.T, self.device = d.cfg, k_ctx, d.T, device
        self.n_episodes = d.n_episodes
        self.n = self.n_episodes * self.T                       # nombre de transitions
        flat = State.from_array(d.states.reshape(-1, d.states.shape[-1]))
        n_frames = flat.n
        frames = np.empty((n_frames, d.cfg.size, d.cfg.size, 3), dtype=np.uint8)
        for lo in range(0, n_frames, 20_000):
            frames[lo:lo + 20_000] = render(flat[lo:lo + 20_000], d.cfg)
        self.frames = torch.from_numpy(frames).to(device)       # (E*(T+1), H, W, 3) uint8
        self.n_frames = n_frames
        self.actions = torch.from_numpy(d.actions.astype(np.int64)).to(device)   # (E, T)
        self.branch = torch.from_numpy(d.branch.astype(np.int64)).to(device)     # (E, T), -1 hors bifurcation
        self.pos = None
        if with_positions:
            self.pos = torch.from_numpy(true_positions(flat, d.cfg).astype(np.float32)).to(device)  # (E*(T+1), 4, 2)

    # ---- échantillonnage sans état -------------------------------------------------------------
    def _gen(self, seed: int, step: int) -> torch.Generator:
        g = torch.Generator(device=self.device)
        g.manual_seed((int(seed) * 1_000_003 + int(step)) % (2 ** 63 - 1))
        return g

    def sample_idx(self, step: int, batch_size: int, seed: int) -> torch.Tensor:
        return torch.randint(0, self.n, (batch_size,), generator=self._gen(seed, step), device=self.device)

    def sample_frames(self, step: int, batch_size: int, seed: int) -> torch.Tensor:
        return torch.randint(0, self.n_frames, (batch_size,), generator=self._gen(seed, step), device=self.device)

    # ---- lots -----------------------------------------------------------------------------------
    @staticmethod
    def to_float(x_nhwc: torch.Tensor) -> torch.Tensor:
        return x_nhwc.permute(*range(x_nhwc.dim() - 3), x_nhwc.dim() - 1, x_nhwc.dim() - 3, x_nhwc.dim() - 2).float() / 127.5 - 1.0

    def batch(self, idx: torch.Tensor) -> dict:
        e, t = idx // self.T, idx % self.T
        base = e * (self.T + 1)
        steps = (t[:, None] + torch.arange(-self.k + 1, 1, device=idx.device)[None, :]).clamp(min=0)
        ctx = self.frames[base[:, None] + steps]                 # (B, k, H, W, 3)
        target = self.frames[base + t + 1]                       # (B, H, W, 3)
        return {"ctx": self.to_float(ctx), "target": self.to_float(target), "action": self.actions[e, t],
                "branch": self.branch[e, t], "idx": idx}

    def frame_batch(self, fidx: torch.Tensor) -> dict:
        out = {"x": self.to_float(self.frames[fidx])}
        if self.pos is not None:
            out["pos"] = self.pos[fidx]
        return out

    def token_batch(self, idx: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Jetons du contexte (B, k, G) et de la cible (B, G) pour les transitions `idx` ; tokens : (E*(T+1), G)."""
        e, t = idx // self.T, idx % self.T
        base = e * (self.T + 1)
        steps = (t[:, None] + torch.arange(-self.k + 1, 1, device=idx.device)[None, :]).clamp(min=0)
        return tokens[base[:, None] + steps].long(), tokens[base + t + 1].long()
