"""Adaptateurs PyTorch -> interface `flou.eval.predictors.Predictor` (NumPy en entrée et en sortie)."""
from __future__ import annotations

import numpy as np
import torch

from ..eval.predictors import Batch
from ..models.registry import build_model
from .runs import RunSpec


class TorchPredictor:
    """Applique un modèle entraîné à un lot (contexte uint8, actions) et renvoie des images float32 dans [0, 1].

    n_steps : nombre de pas d'Euler (diffusion) ; tau : température (jetons) ; amp : fp16 à l'inférence.
    Chaque appel tire une graine du `rng` NumPy fourni par l'évaluateur : les évaluations sont reproductibles.
    max_batch : un lot plus grand est découpé (le planificateur de `flou.eval.planning` envoie des milliers de lignes
    à la fois ; sans découpage la mémoire du GPU pourrait manquer).
    """

    def __init__(self, model, device: str = "cuda", name: str = "modele", n_steps: int | None = None,
                 tau: float = 1.0, oracle: bool = False, amp: bool = True, max_batch: int = 512):
        self.model, self.device, self.name = model.eval(), device, name
        self.n_steps, self.tau, self.oracle = n_steps, tau, oracle
        self.amp = amp and torch.device(device).type == "cuda"
        self.max_batch = max_batch

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        n = len(batch.actions)
        if n <= self.max_batch:
            return self._predict(batch, rng)
        parts = []
        for lo in range(0, n, self.max_batch):
            sl = slice(lo, lo + self.max_batch)
            sub = Batch(batch.idx[sl], batch.ctx[sl], batch.actions[sl], batch.step,
                        None if batch.branch is None else batch.branch[sl])
            parts.append(self._predict(sub, rng))
        return np.concatenate(parts)

    @torch.no_grad()
    def _predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        dev = self.device
        ctx = torch.from_numpy(batch.ctx).to(dev)                                   # (n, k, H, W, 3) uint8
        ctx = ctx.permute(0, 1, 4, 2, 3).float() / 127.5 - 1.0
        act = torch.from_numpy(np.asarray(batch.actions, dtype=np.int64)).to(dev)
        branch = None
        if self.oracle:
            if batch.branch is None:
                raise ValueError("le prédicteur oracle exige la branche (pas 0 seulement : pas de rollout)")
            branch = torch.from_numpy(np.asarray(batch.branch, dtype=np.int64)).to(dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(rng.integers(0, 2 ** 62)))
        kw = {}
        if self.n_steps is not None:
            kw["n_steps"] = self.n_steps
        if self.model.__class__.__name__ == "TokenWorldModel":
            kw["tau"] = self.tau
        with torch.autocast(device_type=torch.device(dev).type, dtype=torch.float16, enabled=self.amp):
            y = self.model.predict(ctx, act, generator=gen, branch=branch, **kw)
        return ((y.float() + 1.0) / 2.0).clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()


def load_run(run_dir, device: str = "cuda"):
    """(RunSpec, modèle en mode évaluation) d'après spec.json et final.pt (poids EMA si l'EMA était active)."""
    spec = RunSpec.load(run_dir)
    model = build_model(spec.model_spec()).to(device)
    state = torch.load(f"{run_dir}/final.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return spec, model.eval()
