"""Interface commune des prédicteurs : modèles PyTorch, baselines et prédicteurs factices.

L'évaluateur ne dépend que de ce contrat, ce qui permet de tester tous les instruments sans
PyTorch (avec des prédicteurs dont on connaît la bonne réponse).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Batch:
    """Un lot de contextes à prédire.

    idx      indices dans le jeu de test (les modèles réels les ignorent ; les prédicteurs
             factices s'en servent pour retrouver la bonne réponse)
    ctx      (n, k, H, W, 3) uint8 : les k dernières images, la plus récente en dernier
    actions  (n,) int : action jouée à l'instant courant
    step     0 pour la prédiction de l'image suivante ; i >= 1 lors d'un rollout autorégressif
    branch   (n,) branche réellement tirée, fournie UNIQUEMENT au pas 0 : seul le régresseur oracle
             (contrôle H0c) a le droit de s'en servir
    """
    idx: np.ndarray
    ctx: np.ndarray
    actions: np.ndarray
    step: int = 0
    branch: np.ndarray | None = None


class Predictor(Protocol):
    name: str

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        """Une image prédite par ligne : (n, H, W, 3), flottants dans [0, 1] (uint8 accepté).
        Un modèle génératif en tire UN échantillon ; un modèle de régression est déterministe."""
        ...


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Quantification 8 bits d'une image prédite (ce que l'on rebouclerait en entrée d'un rollout)."""
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
