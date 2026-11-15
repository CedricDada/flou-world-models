"""Prédicteurs factices dont la bonne réponse est connue : bancs d'essai des instruments.

Chacun incarne un comportement idéal ou fautif que l'évaluateur doit reconnaître :

* `IdealSampler`   tire la branche avec la probabilité q puis rend l'image nette correspondante :
                   avec q = p, c'est le générateur parfait (net ET calibré) ;
* `IdealMean`      rend pA + (1-p)B : le régresseur parfait (floue, MSE au plancher de Bayes) ;
* `ModeCollapse`   rend toujours la branche majoritaire (net, jamais calibré) ;
* `OracleRegressor` rend l'image de la branche réellement tirée (le contrôle H0c) ;
* `Noisy`          ajoute un bruit gaussien à un autre prédicteur ;
* `DropGem`        efface le petit objet (perte de détail, taxonomie (c)).
"""
from __future__ import annotations

import numpy as np

from ..data.dataset import BifurcationSet
from .predictors import Batch


class _Base:
    def __init__(self, bset: BifurcationSet):
        a, b, _ = bset.counterfactuals()
        self.A = a.astype(np.float32) / 255.0
        self.B = b.astype(np.float32) / 255.0
        self.cfg = bset.cfg
        self.branch = bset.branch
        s_a, _ = bset.next_states()
        self.gem = np.stack([s_a.gx, s_a.gy], axis=1)


class IdealSampler(_Base):
    def __init__(self, bset: BifurcationSet, q: float | None = None, name: str | None = None):
        super().__init__(bset)
        self.q = bset.cfg.p if q is None else q
        self.name = name or f"echantillonneur-ideal(q={self.q:g})"

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        pick_a = rng.random(len(batch.idx)) < self.q
        return np.where(pick_a[:, None, None, None], self.A[batch.idx], self.B[batch.idx])


class IdealMean(_Base):
    name = "regresseur-ideal"

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        p = self.cfg.p
        return p * self.A[batch.idx] + (1 - p) * self.B[batch.idx]


class ModeCollapse(_Base):
    def __init__(self, bset: BifurcationSet):
        super().__init__(bset)
        self.name = "effondrement-de-mode"

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        major_a = self.cfg.p >= 0.5
        return (self.A if major_a else self.B)[batch.idx]


class OracleRegressor(_Base):
    name = "regresseur-oracle"

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        is_a = self.branch[batch.idx] == 1
        return np.where(is_a[:, None, None, None], self.A[batch.idx], self.B[batch.idx])


class Noisy:
    def __init__(self, inner, sigma: float):
        self.inner, self.sigma = inner, sigma
        self.name = f"{inner.name}+bruit({sigma:g})"

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        y = np.asarray(self.inner.predict(batch, rng), dtype=np.float32)
        y = y + rng.normal(0.0, self.sigma, y.shape).astype(np.float32)
        return np.clip(np.round(y * 255.0) / 255.0, 0.0, 1.0)


class DropGem(_Base):
    """Rend l'image du générateur parfait, mais sans le petit objet."""

    def __init__(self, bset: BifurcationSet, q: float | None = None):
        super().__init__(bset)
        self.inner = IdealSampler(bset, q)
        self.name = "perte-du-petit-objet"

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        y = np.array(self.inner.predict(batch, rng), copy=True)
        bg = np.asarray(self.cfg.color_bg, dtype=np.float32) / 255.0
        g = self.cfg.gem_size
        for j, i in enumerate(batch.idx):
            gx, gy = self.gem[i]
            y[j, gy:gy + g, gx:gx + g] = bg
        return y


class TruthRollout:
    """Rejoue une vérité terrain de rollout (N, H, 64, 64, 3) uint8 : sert à tester l'évaluation à
    l'horizon 16. Si `drift_after` est donné, à partir de ce pas toutes les images sont translatées
    de 6 pixels vers la droite : l'agent n'est alors plus à moins de 3 pixels de la vérité (erreur
    cumulative simulée, de sorte que la précision de suivi doit chuter à partir de ce pas)."""

    def __init__(self, frames: np.ndarray, drift_after: int | None = None, name: str = "verite-terrain"):
        self.frames = frames
        self.drift_after = drift_after
        self.name = name

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        y = self.frames[batch.idx, batch.step].astype(np.float32) / 255.0
        if self.drift_after is not None and batch.step >= self.drift_after:
            y = np.roll(y, 6, axis=2)
        return y
