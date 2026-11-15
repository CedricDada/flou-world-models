"""Validation de l'instrument AVANT de mesurer un modèle (contrôle H0a et sonde analytique).

* règle ρmax sur des images pures bruitées (niveau de bruit `noise_sigma`) : plus petit seuil classant au
  moins 99,5 % des images pures correctement ;
* contrôle H0a sans entraînement : erreur moyenne sur α̂ < 0,05 et plus de 99 % d'images pures bien classées ;
* sensibilité : la même règle à 0,5 x, 1 x et 2 x ρmax, pour que l'analyse puisse montrer à quel point la
  part de « autre » dépend du seuil ;
* sonde analytique : RMSE par objet sur de vraies images (référence de plancher pour la sonde apprise).

Résultat écrit dans reports/instrument.json ; `scripts/evaluate.py` y lit ρmax.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.dataset import BifurcationSet
from ..game.render import branch_delta, render
from .classifier import h0a_check, tune_rho_max
from .probe import AnalyticProbe, probe_report, true_positions


def validate_instrument(test_files: list[str | Path], noise_sigma: float = 0.01, seed: int = 0,
                        quantile: float = 0.995) -> dict:
    """`test_files` : jeux de test de bifurcation (un par valeur de p). ρ est normalisé par δ : on règle
    ρmax sur l'ensemble des paires (A, B) de tous les jeux."""
    sets = [BifurcationSet.load(p) for p in test_files]
    A = np.concatenate([s.counterfactuals()[0] for s in sets])
    B = np.concatenate([s.counterfactuals()[1] for s in sets])
    gem = np.concatenate([np.stack([n.gx, n.gy], axis=1) for n in (s.next_states()[0] for s in sets)])
    delta = branch_delta(A, B)
    rng = np.random.default_rng(seed)
    rho_max = tune_rho_max(A, B, noise_sigma, rng, quantile=quantile)
    h0a = h0a_check(A, B, gem, rho_max, noise_sigma, seed=seed + 1)
    sens = {f"x{m:g}": h0a_check(A, B, gem, rho_max * m, noise_sigma, seed=seed + 1)
            for m in (0.5, 1.0, 2.0)}
    probe = {}
    for s, f in zip(sets, test_files):
        st = s.last_state()
        rep = probe_report(AnalyticProbe(s.cfg)(render(st, s.cfg)), true_positions(st, s.cfg))
        probe[f"p={s.cfg.p:.2f}"] = rep
    return {
        "noise_sigma": noise_sigma, "rho_max": rho_max, "quantile": quantile, "n_pairs": int(len(A)),
        "delta_range": [float(delta.min()), float(delta.max())],
        "h0a": h0a,
        "sensitivity": {k: {m: v[m] for m in ("rho_max", "pure_ok", "alpha_mae_mix", "broken_flagged_other", "pass")
                            if m in v} for k, v in sens.items()},
        "analytic_probe": probe,
        "files": [str(f) for f in test_files],
        "avertissement_taille": small_sample_warning(int(len(A))),
    }


MIN_PAIRS_H0A = 600


def small_sample_warning(n_pairs: int) -> str | None:
    """H0a est un test statistique : ρmax est réglé pour classer 99,5 % des images pures, le contrôle exige plus de
    99 %. La marge est de 0,5 point : avec peu d'images pures, quelques erreurs de plus que la moyenne suffisent à
    faire échouer le contrôle sans que l'instrument soit en cause. Mesuré sur les jeux de test du jeu (graines
    0 à 39 / 0 à 29 de la validation, bruit σ = 0,01) : 14 échecs sur 40 à 120 paires, 0 sur 30 à 450 paires ; à
    3 000 paires (1 000 états x 3 valeurs de p) le contrôle réussit avec 99,47 % d'images pures bien classées."""
    if n_pairs >= MIN_PAIRS_H0A:
        return None
    return (f"attention : {n_pairs} paires seulement (< {MIN_PAIRS_H0A}). Un échec de H0a peut venir de la taille de "
            "l'échantillon (marge de 0,5 point entre le réglage à 99,5 % et le seuil de 99 %) ; relancer avec plus "
            "d'états de test avant de conclure que l'instrument est défaillant.")


def save_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return path


def load_rho_max(path: str | Path) -> float:
    return float(json.loads(Path(path).read_text(encoding="utf-8"))["rho_max"])
