"""Validation de l'extension « décision » AVANT de mesurer un modèle (contrôles H0e et H0f). NumPy seulement.

Même principe que `flou.eval.instrument` pour H0a : on vérifie d'abord l'instrument sur des cas dont la bonne réponse
est connue. Trois parties, écrites dans reports/decision_instrument.json :

* H0e   le lecteur de contact retrouve l'événement `hit_enemy` du jeu sur de vraies images (accord >= 99 %) ;
* table « attendu si les modèles étaient parfaits » : la chaîne du niveau 1 appliquée à des prédicteurs factices
        (régression idéale, échantillonneur net calibré, échantillonneur net avec le biais d'Euler du cas jouet) ;
        à k = 1 on la compare à la valeur théorique : c'est le test de bout en bout de la mesure ;
* H0f   le planificateur fonctionne quand on lui donne le VRAI jeu comme modèle (référence de PlaNet, Hafner et al.
        2019, arXiv:1811.04551, Table 1 : planificateur + vrai simulateur).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..data.dataset import BifurcationSet
from ..models import edm_math as E
from . import decision as D
from . import planning as P


def euler_q(p: float, n_steps: int = 3, delta: float = 15.0) -> float:
    """Probabilité de la branche A produite par n pas d'Euler avec un débruiteur EXACT, sur le cas jouet à deux points
    (`flou.models.edm_math`, docs/EXPERIMENTS.md §5). Sert de biais de calibration plausible, pas de prédiction."""
    if p in (0.0, 1.0):
        return p
    return float(E.toy_p_hat(E.euler_sample_toy(p, n_steps, 100_000, sigma_max=80.0 / delta))[0])


def expected_table(sub: BifurcationSet, ks=(1, 3, 5), m: int = 8, seed: int = 0) -> list[dict]:
    """Erreur du niveau 1 pour trois prédicteurs factices parfaits ou presque. `sub` : états déjà sélectionnés."""
    p, kmax = sub.cfg.p, max(ks)
    q = euler_q(p)
    cases = [("régression idéale (image moyenne)", D.BranchRollout(sub, kmax), 1, None),
             (f"échantillonneur net calibré, m = {m}", D.BranchRollout(sub, kmax, q=p), m, p),
             (f"échantillonneur net, biais d'Euler n = 3 (q = {q:.3f}), m = {m}", D.BranchRollout(sub, kmax, q=q), m, q)]
    rows = []
    for label, pred, mm, qq in cases:
        res = D.evaluate_risk(pred, sub, ks, m=mm, seed=seed)
        row = {"prédicteur factice": label}
        for k in ks:
            row[f"doux_k{k}"] = res["per_k"][str(k)]["soft"]["amb"]["mean"]
            row[f"seuil_k{k}"] = res["per_k"][str(k)]["hard"]["amb"]["mean"]
        row["théorie_k1_seuil"] = (D.expected_error_mean(p, "hard") if qq is None
                                   else D.expected_error_sampler(p, qq, mm))
        rows.append(row)
    return rows


def validate_decision(bset: BifurcationSet, ks=(1, 3, 5), m: int = 8, seed: int = 0, planner: bool = True,
                      pc: P.PlanConfig = P.PlanConfig(), n_games: int = 50, game_steps: int | None = None,
                      progress=None) -> dict:
    cfg = bset.cfg
    idx = D.select_states(bset, ks, seed=seed)
    sub = bset.subset(idx)
    truth = D.true_costs(bset, max(ks))
    rep = {
        "p": cfg.p, "n_states": len(bset), "n_selected": len(sub), "ks": list(ks), "m": m,
        "ambiguous_share": {f"k={k}": float(truth["ambiguous"][:, k - 1].mean()) for k in ks},
        "h0e": D.check_reader(bset),
        "expected": expected_table(sub, ks, m, seed),
        "h0f": P.check_planner(cfg, pc, n_games, seed, game_steps, progress=progress) if planner else None,
    }
    return rep


def save_report(rep: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    return path
