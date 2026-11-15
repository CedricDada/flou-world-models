"""RÉPÉTITION GÉNÉRALE avec des prédicteurs FACTICES (aucun entraînement, aucun résultat scientifique).

Fabrique un dossier `runs/` complet pour les 25 entraînements du plan en remplaçant chaque modèle par un
prédicteur dont on connaît le comportement, afin de tester de bout en bout l'évaluateur, l'analyse, les
verdicts et les figures SANS GPU :

    regression  -> régresseur idéal (flou, MSE au plancher de Bayes)
    oracle      -> régresseur qui connaît la branche tirée
    diffusion   -> n = 1 : régresseur idéal ; n >= 2 : générateur net dont p̂ suit le biais d'Euler du cas jouet
    tokens      -> générateur net calibré qui perd le petit objet avec une probabilité décroissante avec K

Les probabilités de perte du petit objet (0,50 / 0,25 / 0,10 pour K = 64 / 256 / 1024) et les décrochages de
rollout sont INVENTÉS : ils servent à vérifier que l'analyse sait reconnaître un effet, pas à en mesurer un.
Tout ce que cette répétition écrit porte la mention « SIMULATION ».

Extension « décision » (docs/EXTENSION_DECISION.md), runs du cœur seulement :

    niveau 1    regression -> image moyenne exacte ; diffusion n = 1 -> idem ; diffusion n = 3 -> générateur net avec
                le biais d'Euler du cas jouet ; tokens -> générateur net calibré
    niveau 2    diffusion -> le vrai jeu comme modèle, branche tirée avec le biais d'Euler ;
                regression -> image moyenne dont l'ennemi DISPARAÎT dès le premier pas imaginé (effet INVENTÉ et
                volontairement énorme, pour vérifier que l'analyse sait confirmer H5a ; avec une image moyenne
                exacte, le planificateur fait au contraire très bien : voir docs/EXTENSION_DECISION.md §3)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.dataset import BifurcationSet
from ..models import edm_math as E
from ..train.runs import RunSpec
from . import evaluate as ev
from . import fakes
from . import rollout as ro
from .arrays import pack_arrays, variant_key
from .predictors import Batch

GEM_LOSS_BY_K = {64: 0.50, 256: 0.25, 1024: 0.10}


def drift_after(spec: RunSpec) -> int | None:
    """Pas à partir duquel le rollout simulé décroche (None : jamais). Inventé, voir l'en-tête du module."""
    if spec.family == "regression":
        return 9
    if spec.family == "diffusion":
        return 11
    if spec.family == "tokens":
        return {64: 8, 256: 8, 1024: None}[spec.K]
    return None


class RandomGemLoss:
    """Efface le petit objet avec une probabilité donnée (perte de détail simulée)."""

    def __init__(self, inner, bset: BifurcationSet, prob: float):
        self.inner, self.prob, self.cfg = inner, prob, bset.cfg
        s_a, _ = bset.next_states()
        self.gem = np.stack([s_a.gx, s_a.gy], axis=1)
        self.name = f"{inner.name}+perte-objet({prob:g})"

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        y = np.array(self.inner.predict(batch, rng), copy=True)
        bg = np.asarray(self.cfg.color_bg, dtype=np.float32) / 255.0
        g = self.cfg.gem_size
        drop = rng.random(len(y)) < self.prob
        for j in np.nonzero(drop)[0]:
            gx, gy = self.gem[batch.idx[j]]
            y[j, gy:gy + g, gx:gx + g] = bg
        return y


def fake_predictor(spec: RunSpec, bset: BifurcationSet, n_steps: int | None = None, delta: float = 15.0):
    p = bset.cfg.p
    if spec.family == "regression" or (spec.family == "diffusion" and n_steps == 1):
        return fakes.IdealMean(bset)
    if spec.family == "oracle":
        return fakes.OracleRegressor(bset)
    if spec.family == "diffusion":
        if p in (0.0, 1.0):
            return fakes.IdealSampler(bset, q=p)
        q = E.toy_p_hat(E.euler_sample_toy(p, n_steps, 100_000, sigma_max=80.0 / delta))[0]
        return fakes.IdealSampler(bset, q=0.0 if np.isnan(q) else q, name=f"euler(n={n_steps})")
    if spec.family == "tokens":
        return RandomGemLoss(fakes.IdealSampler(bset, q=p), bset, GEM_LOSS_BY_K[spec.K])
    raise ValueError(spec.family)


def simulate_run(spec: RunSpec, bset: BifurcationSet, rho_max: float, runs_dir: str | Path, n_list=(1, 3, 5, 20),
                 rollout_n: int = 100, horizon: int = 16, gpu_hours: float = 1.5) -> dict:
    run_dir = Path(runs_dir) / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    spec.save(run_dir)
    seed = 1000 * spec.seed + int(100 * spec.p) + (7 if spec.family == "diffusion" else 0)
    variants = ([(f"n={n}", n) for n in n_list] if spec.family == "diffusion"
                else [("tau=1", None)] if spec.family == "tokens" else [("", None)])
    out = {"run_id": spec.run_id, "spec": spec.to_dict(), "rho_max": rho_max, "n_states": len(bset),
           "probe": "SIMULATION", "variants": []}
    arrays: dict = {}
    sub = bset.subset(np.arange(min(rollout_n, len(bset))))
    truth, _ = ro.rollout_truth(sub, horizon, seed=seed)
    for name, n in variants:
        pred = fake_predictor(spec, bset, n)
        res = ev.evaluate_predictor(pred, bset, rho_max, seed=seed, keep_images=True)
        arrays.update(pack_arrays(res, variant_key(name)))
        entry = {"variant": name, "summary": ev.summarize(res), "chk_h0b": ev.check_h0b(res),
                 "chk_blur5": ev.check_blur_below(res), "seconds": 0.0}
        if spec.family != "oracle":
            tr = fakes.TruthRollout(truth, drift_after=drift_after(spec))
            entry["rollout"] = ro.evaluate_rollout(tr, sub, horizon, seed=seed)
        out["variants"].append(entry)
    if spec.family == "tokens":
        loss = GEM_LOSS_BY_K[spec.K]
        out["vq"] = {"gem_recall": 1.0 - loss * 0.5, "recon_mse": 0.001, "perplexity_ema": 0.6 * spec.K, "K": spec.K}
    out["SIMULATION"] = True
    np.savez_compressed(run_dir / "eval_arrays.npz", **arrays)
    (run_dir / "eval.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (run_dir / "compute.json").write_text(json.dumps({"gpu": "SIMULATION", "gpu_hours": gpu_hours,
                                                      "parameters": 6_000_000}), encoding="utf-8")
    return out


def simulate_plan(plan: list[RunSpec], data_root: str | Path, runs_dir: str | Path, rho_max: float,
                  n_states: int = 300, seed: int = 0, **kw) -> list[dict]:
    """Simule tous les runs d'un plan. Construit un jeu de test (n_states états) par valeur de p."""
    from ..data.dataset import build_bifurcation_set
    from ..game import GameConfig
    from ..train.runs import data_dir_for
    sets = {}
    for p in sorted({r.p for r in plan}):
        cfg = GameConfig().with_p(p)
        bs = build_bifurcation_set(cfg, n=n_states, seed=seed)
        bs.save(data_dir_for(data_root, p) / "test_bifurcation.npz", {"config": cfg.to_dict()})
        sets[p] = bs
    return [simulate_run(r, sets[r.p], rho_max, runs_dir, **kw) for r in plan]


# ---------------------------------------------------------------------------------------
# Extension « décision » : fichiers decision.json et planning.json simulés
# ---------------------------------------------------------------------------------------
def simulate_decision(spec: RunSpec, sub: BifurcationSet, runs_dir: str | Path, ks=(1, 3, 5), m: int = 8) -> dict:
    """Niveau 1 simulé pour un run du cœur. `sub` : états déjà sélectionnés (`decision.select_states`)."""
    from . import decision as D
    from .decision_instrument import euler_q
    p, kmax = sub.cfg.p, max(ks)
    if spec.family == "regression":
        variants = [("", D.BranchRollout(sub, kmax), 1)]
    elif spec.family == "diffusion":
        variants = [("n=1", D.BranchRollout(sub, kmax), m),
                    ("n=3", D.BranchRollout(sub, kmax, q=euler_q(p, 3), name="euler(n=3)"), m)]
    elif spec.family == "tokens":
        variants = [("tau=1", D.BranchRollout(sub, kmax, q=p), m)]
    else:
        raise ValueError(spec.family)
    out = {"run_id": spec.run_id, "spec": spec.to_dict(), "n_states": len(sub), "SIMULATION": True, "variants": []}
    for name, pred, mm in variants:
        res = D.evaluate_risk(pred, sub, ks, m=mm, seed=1000 * spec.seed + 13)
        out["variants"].append({"variant": name, "risk": res, "seconds": 0.0})
    run_dir = Path(runs_dir) / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "decision.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def simulate_planning(spec: RunSpec, cfg, runs_dir: str | Path, pc=None, n_games: int = 8, seed: int = 0,
                      game_steps: int | None = None) -> dict:
    """Niveau 2 simulé pour un run du cœur (régression ou diffusion). L'effet planté côté régression est INVENTÉ."""
    from dataclasses import replace as _replace
    from . import planning as P
    from .decision_instrument import euler_q
    pc = pc or P.PlanConfig()
    if spec.family == "regression":
        name, pred, pc = "", P.SimulatorModel(cfg, "mean", fade=0.0), _replace(pc, samples=1)
    elif spec.family == "diffusion":
        name, pred = "n=3", P.SimulatorModel(cfg, q=euler_q(cfg.p, 3))
    else:
        raise ValueError(spec.family)
    res = P.play(pred, cfg, pc, n_games, seed, plan_seed=1000 * (spec.seed + 1) + seed, steps=game_steps)
    out = {"run_id": spec.run_id, "spec": spec.to_dict(), "SIMULATION": True,
           "variants": [{"variant": name, "play": res}]}
    run_dir = Path(runs_dir) / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "planning.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out
