"""Baselines triviales qui situent l'échelle des erreurs (dossier §1.6, cours chap. 9)."""
from __future__ import annotations

import numpy as np

from .predictors import Batch


class CopyLast:
    """« Image suivante = image courante » : ne modélise ni le mouvement ni la bifurcation."""
    name = "copie-derniere-image"

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        return batch.ctx[:, -1].astype(np.float32) / 255.0
