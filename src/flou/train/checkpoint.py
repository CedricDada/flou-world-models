"""Checkpoints reprenables (cours, chap. 8) : modèle, optimiseur, état du GradScaler, états aléatoires,
position dans l'entraînement et temps de calcul déjà consommé. Écriture atomique (fichier temporaire puis
renommage) : une interruption pendant l'écriture ne corrompt pas le checkpoint précédent."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_atomic(obj: dict, path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load(path: str | Path, device: str = "cpu") -> dict:
    # weights_only=False : le checkpoint contient des états aléatoires NumPy ; fichiers produits par ce dépôt seulement
    return torch.load(path, map_location=device, weights_only=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
