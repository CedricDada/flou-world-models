"""Évaluation complète d'un run entraîné : jeu de test de bifurcation, variantes d'échantillonnage,
rollouts de 16 pas, rappel du petit objet, planches d'exemples. Écrit eval.json et eval_arrays.npz dans le
dossier du run (les tableaux par échantillon servent à re-seuiller ρmax sans réévaluer).

Variantes évaluées (les mêmes états de test pour toutes) :
    regression, oracle   une variante (sortie déterministe) ;
    diffusion            n in {1, 3, 5, 20} pas d'Euler, avec LE MÊME débruiteur (comparaisons appariées) ;
    tokens               température 1, sans top-k ; en plus, rappel du petit objet après simple
                         reconstruction par le VQ-VAE (borne supérieure du rappel de la famille C).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import BifurcationSet
from ..eval import evaluate as ev
from ..eval import rollout as ro
from ..eval.arrays import pack_arrays, variant_key
from ..eval.probe import AnalyticProbe
from ..eval.recall import gem_recall
from ..game.viz import contact_sheet
from .predictors import TorchPredictor, load_run
from .runs import RunSpec, test_file


def variants_for(spec: RunSpec, n_list=(1, 3, 5, 20)) -> list[tuple[str, dict]]:
    if spec.family == "diffusion":
        return [(f"n={n}", {"n_steps": n}) for n in n_list]
    if spec.family == "tokens":
        return [("tau=1", {"tau": 1.0})]
    return [("", {})]


def _sheet(pred, bset: BifurcationSet, path: Path, n_states: int = 6, n_samples: int = 3) -> None:
    sub = bset.subset(np.arange(n_states))
    imgs, sidx = ev.predict_set(pred, sub, n_samples=n_samples, batch_size=n_states, seed=0)
    a, b, _ = sub.counterfactuals()
    last = (sub.contexts()[:, -1])
    rows, labels = [], []
    for i in range(n_states):
        samples = [np.round(imgs[sidx == i][j] * 255).astype(np.uint8) for j in range(n_samples)]
        rows.append([last[i], a[i], b[i], *samples])
        labels.append(["contexte t", "A", "B", *[f"tirage {j + 1}" for j in range(n_samples)]])
    contact_sheet(rows, path, scale=3, labels=labels)


def vq_capacity(model, bset: BifurcationSet, device: str) -> dict:
    """Rappel du petit objet et MSE après reconstruction PAR LE SEUL VQ-VAE des vraies images suivantes."""
    real = bset.realized_images()
    s_a, _ = bset.next_states()
    out = []
    for i in range(0, len(real), 250):
        x = torch.from_numpy(real[i:i + 250]).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        out.append(((model.reconstruct(x).float() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
    rec = np.concatenate(out)
    q = np.clip(np.round(rec * 255), 0, 255).astype(np.uint8)
    ok = gem_recall(q, s_a.gx, s_a.gy, bset.cfg)
    mse = float(((rec - real.astype(np.float32) / 255.0) ** 2).mean())
    usage = model.vq.usage.detach().float().cpu().numpy()
    pi = usage / usage.sum()
    ppl = float(np.exp(-(pi[pi > 0] * np.log(pi[pi > 0])).sum()))
    return {"gem_recall": float(ok.mean()), "recon_mse": mse, "perplexity_ema": ppl, "K": int(model.vq.cfg.K)}


def evaluate_run(run_dir: str | Path, data_root: str | Path, rho_max: float, device: str = "cuda",
                 n_list=(1, 3, 5, 20), n_samples: int = 1, rollout_n: int = 250, horizon: int = 16,
                 probe=None, seed: int = 0, amp: bool = True, verbose: bool = True) -> dict:
    run_dir = Path(run_dir)
    spec, model = load_run(run_dir, device)
    bset = BifurcationSet.load(test_file(data_root, spec.p))
    probe = probe or AnalyticProbe(bset.cfg)
    result = {"run_id": spec.run_id, "spec": spec.to_dict(), "rho_max": rho_max, "n_states": len(bset),
              "probe": type(probe).__name__, "variants": []}
    arrays: dict[str, np.ndarray] = {}
    for name, kw in variants_for(spec, n_list):
        t0 = time.time()
        pred = TorchPredictor(model, device, name=f"{spec.run_id}[{name}]", oracle=spec.family == "oracle",
                              amp=amp, **kw)
        res = ev.evaluate_predictor(pred, bset, rho_max, n_samples=n_samples, seed=seed, keep_images=True,
                                    token_grid=spec.grid if spec.family == "tokens" else 8)
        entry = {"variant": name, "summary": ev.summarize(res), "chk_h0b": ev.check_h0b(res),
                 "chk_blur5": ev.check_blur_below(res)}
        if spec.family != "oracle" and rollout_n > 0:
            sub = bset.subset(np.arange(min(rollout_n, len(bset))))
            entry["rollout"] = ro.evaluate_rollout(pred, sub, horizon, seed=seed, probe=probe)
        entry["seconds"] = time.time() - t0
        result["variants"].append(entry)
        key = variant_key(name)
        arrays.update(pack_arrays(res, key))
        try:
            _sheet(pred, bset, run_dir / f"samples_{key.replace('=', '')}.png")
        except ImportError:
            pass
        if verbose:
            print(ev.one_line(res), f"({entry['seconds']:.0f} s)")
    if spec.family == "tokens":
        result["vq"] = vq_capacity(model, bset, device)
    (run_dir / "eval.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    np.savez_compressed(run_dir / "eval_arrays.npz", **arrays)
    return result


# ---------------------------------------------------------------------------------------
# Extension « décision » (docs/EXTENSION_DECISION.md) : niveau 1 (H4) et niveau 2 (H5)
# ---------------------------------------------------------------------------------------
def decision_file(data_root: str | Path, p: float) -> Path:
    from .runs import data_dir_for
    path = data_dir_for(data_root, p) / "test_decision.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} manquant : lancer scripts/validate_decision.py --p {p}")
    return path


def decision_variants(spec: RunSpec, n_list=(1, 3)) -> list[tuple[str, dict]]:
    """Variantes mesurées par l'extension. Diffusion : n = 3 est la variante des hypothèses H4a, H4b, H5a ; n = 1 sert
    de repère (elle devrait se comporter comme la régression)."""
    if spec.family == "diffusion":
        return [(f"n={n}", {"n_steps": n}) for n in n_list]
    if spec.family == "tokens":
        return [("tau=1", {"tau": 1.0})]
    if spec.family == "regression":
        return [("", {})]
    raise ValueError(f"famille {spec.family} : pas de mesure de décision (le régresseur oracle reçoit la branche)")


def evaluate_decision_run(run_dir: str | Path, data_root: str | Path, device: str = "cuda", ks=(1, 3, 5), m: int = 8,
                          n_list=(1, 3), seed: int = 0, amp: bool = True, batch_size: int = 250,
                          verbose: bool = True) -> dict:
    """Niveau 1 : erreur d'estimation du coût de contact (H4a, H4b, H4c). Écrit decision.json dans le dossier du run.

    Régression : sortie déterministe, un seul rollout ; générateurs : m tirages par état (m = 8 par défaut ; PETS,
    arXiv:1805.12114 §5.1, en utilise 20). Les états mesurés sont ceux à risque ambigu, plus autant d'états non
    ambigus (`flou.eval.decision.select_states`)."""
    from ..eval import decision as D
    run_dir = Path(run_dir)
    spec, model = load_run(run_dir, device)
    bset = BifurcationSet.load(decision_file(data_root, spec.p))
    sub = bset.subset(D.select_states(bset, ks, seed=seed))
    out = {"run_id": spec.run_id, "spec": spec.to_dict(), "n_states_file": len(bset), "n_states": len(sub),
           "variants": []}
    for name, kw in decision_variants(spec, n_list):
        t0 = time.time()
        pred = TorchPredictor(model, device, name=f"{spec.run_id}[{name}]", amp=amp, **kw)
        res = D.evaluate_risk(pred, sub, ks, m=1 if spec.family == "regression" else m, seed=seed,
                              batch_size=batch_size)
        out["variants"].append({"variant": name, "risk": res, "seconds": time.time() - t0})
        if verbose:
            print(D.risk_line(res), f"({time.time() - t0:.0f} s)")
    (run_dir / "decision.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def plan_run(run_dir: str | Path, data_root: str | Path, device: str = "cuda", pc=None, n_games: int = 50,
             seed: int = 0, n_steps: int = 3, game_steps: int | None = None, amp: bool = True,
             verbose: bool = True, save: bool = True) -> dict:
    """Niveau 2 : le planificateur joue dans le vrai jeu avec ce modèle comme simulateur (H5a, H5b). Écrit
    planning.json. Les parties ne dépendent que de `seed` : tous les runs jouent les mêmes (comparaison appariée) ;
    les suites d'actions tirées par le planificateur dépendent aussi de la graine du run.

    Régression : m = 1 (sortie déterministe : c'est la « propagation de l'espérance » de PETS §5.2) ; générateurs :
    `pc.samples` tirages par suite candidate. Diffusion : n = `n_steps` pas d'Euler (3 : la variante de H5a)."""
    from dataclasses import replace as _replace
    from ..eval import planning as P
    run_dir = Path(run_dir)
    spec, model = load_run(run_dir, device)
    cfg = BifurcationSet.load(test_file(data_root, spec.p)).cfg
    pc = pc or P.PlanConfig()
    pc = _replace(pc, k_ctx=spec.k_ctx, samples=1 if spec.family == "regression" else pc.samples)
    name, kw = [(n, k) for n, k in decision_variants(spec, (n_steps,))][0]
    pred = TorchPredictor(model, device, name=f"{spec.run_id}[{name}]", amp=amp, **kw)

    def progress(t, total, sec):
        if verbose and (t in (1, 5) or t % 20 == 0 or t == total):
            print(f"   décision {t}/{total} : {sec:.0f} s écoulées, fin estimée dans {sec / t * (total - t) / 60:.1f} min",
                  flush=True)

    res = P.play(pred, cfg, pc, n_games, seed, plan_seed=1000 * (spec.seed + 1) + seed, steps=game_steps,
                 progress=progress)
    out = {"run_id": spec.run_id, "spec": spec.to_dict(), "variants": [{"variant": name, "play": res}]}
    if save:                                    # save=False : chronométrage, on n'écrit rien
        (run_dir / "planning.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    if verbose:
        print(P.play_line(res))
    return out
