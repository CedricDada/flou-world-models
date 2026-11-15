"""Tableaux par échantillon sauvegardés avec chaque évaluation (eval_arrays.npz). NumPy seulement.

Pour chaque variante (clé `n=3`, `tau=1`, `main`...) :
    {clé}__alpha, __rho, __label, __recall, __state   un élément par image prédite : permettent de re-seuiller ρmax
                                                      et de refaire les tables sans réévaluer le modèle ;
    {clé}__img_idx, {clé}__img                        au plus `per_class` images PAR CLASSE AUTOMATIQUE (tirage
                                                      uniforme dans la classe) : réservoir de la relecture manuelle ;
    {clé}__ab_img                                     les `n_ab` premiers états, premier tirage : mêmes contextes pour
                                                      tous les modèles d'un même p, pour le test A/B en aveugle.
Les images sont en uint8 (H, W, 3).
"""
from __future__ import annotations

import numpy as np


def _to_u8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def pack_arrays(res: dict, key: str, per_class: int = 25, n_ab: int = 40, seed: int = 0,
                n_classes: int = 4) -> dict[str, np.ndarray]:
    ps = res["per_sample"]
    out = {f"{key}__{k}": np.asarray(ps[k]) for k in ("alpha", "rho", "label", "recall", "state")}
    imgs = res.get("images")
    if imgs is None:
        return out
    label = np.asarray(ps["label"])
    rng = np.random.default_rng(seed)
    pick = [rng.choice(np.nonzero(label == c)[0], min(per_class, int((label == c).sum())), replace=False)
            for c in range(n_classes) if (label == c).any()]
    idx = np.sort(np.concatenate(pick)) if pick else np.zeros(0, dtype=int)
    out[f"{key}__img_idx"] = idx.astype(np.int64)
    out[f"{key}__img"] = _to_u8(imgs[idx])
    n_states = int(res["n_states"])
    first = np.asarray(ps["state"][:n_states])
    if len(first) == n_states and (first == np.arange(n_states)).all():       # premier tirage rangé par état
        out[f"{key}__ab_img"] = _to_u8(imgs[:min(n_ab, n_states)])
    return out


def variant_key(variant: str) -> str:
    return variant or "main"
