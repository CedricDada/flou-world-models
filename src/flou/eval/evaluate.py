"""Évaluation d'un prédicteur sur le jeu de test de bifurcation.

Pour chaque état de bifurcation du jeu de test (identique pour tous les modèles), le prédicteur
produit l'image suivante ; on la classe avec le classifieur contrefactuel (nette-A, nette-B, floue,
autre), puis on mesure :

* la **netteté** (part de floues) et la **calibration** p̂ = N_A / (N_A + N_B) avec son intervalle de
  Clopper-Pearson, à comparer à p ;
* la **MSE** à la fois contre l'image réellement tirée et en espérance sur les deux branches, rapportée
  au plancher de Bayes p(1-p)δ² (contrôle H0b) ;
* le **rappel du petit objet**, les objets hallucinés, la part de pixels hors palette et l'excès de
  discontinuité sur les frontières de blocs (artefacts de jetons).

Deux lectures de la MSE, pourquoi : contre l'image tirée (« réalisée »), un modèle qui devinerait la
branche passerait SOUS le plancher (fuite d'information) ; mais ce rapport est bruité (≈ 3 % à
N = 1 000, p = 0,3), si bien qu'un régresseur parfait passe sous 1,00 une fois sur deux. En espérance
sur les deux branches (p||ŷ-A||² + (1-p)||ŷ-B||²), le bruit d'étiquette disparaît et le rapport vaut
exactement 1 + ||ŷ-m||²/plancher : il mesure le seul sous-apprentissage. On rapporte les deux.
"""
from __future__ import annotations

import numpy as np

from ..data.dataset import BifurcationSet
from .classifier import CLASS_A, CLASS_B, CLASS_NAMES, EPS, alpha_rho, as_float, classify
from .predictors import Batch, Predictor, to_uint8
from .recall import (block_edge_excess, gem_hallucination, gem_recall, off_palette_fraction)
from .stats import p_hat, proportion


def bayes_floor(delta: np.ndarray, p: float) -> np.ndarray:
    """Plancher de Bayes de l'erreur quadratique (somme sur les pixels, échelle 0-1) : p(1-p)δ²."""
    return p * (1.0 - p) * np.asarray(delta, dtype=np.float64) ** 2


def predict_set(pred: Predictor, bset: BifurcationSet, n_samples: int = 1, batch_size: int = 250,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Applique le prédicteur à tout le jeu. Renvoie (images (n_samples * N, H, W, 3) float32,
    indices d'états (n_samples * N)). Les images sont rangées échantillon par échantillon."""
    ctx_all = bset.contexts()
    rng = np.random.default_rng(seed)
    outs, idxs = [], []
    n = len(bset)
    for _ in range(n_samples):
        for lo in range(0, n, batch_size):
            idx = np.arange(lo, min(n, lo + batch_size))
            y = pred.predict(Batch(idx, ctx_all[idx], bset.actions[idx], 0, bset.branch[idx]), rng)
            outs.append(as_float(y).astype(np.float32))
            idxs.append(idx)
    return np.concatenate(outs), np.concatenate(idxs)


def evaluate_predictor(pred: Predictor, bset: BifurcationSet, rho_max: float, *, n_samples: int = 1,
                       batch_size: int = 250, seed: int = 0, eps: float = EPS, conf: float = 0.95,
                       keep_images: bool = False, token_grid: int = 8) -> dict:
    """Toutes les mesures d'un prédicteur sur le jeu de test. Renvoie un dictionnaire sérialisable
    (les tableaux par exemple sont sous `per_sample`, retirés par `summarize`)."""
    cfg = bset.cfg
    p = cfg.p
    img_a, img_b, delta = bset.counterfactuals()
    s_a, _ = bset.next_states()
    fa, fb = as_float(img_a), as_float(img_b)
    real = np.where((bset.branch == 1)[:, None, None, None], fa, fb)

    imgs, sidx = predict_set(pred, bset, n_samples, batch_size, seed)
    N = len(imgs)
    a, b = fa[sidx], fb[sidx]
    alpha, rho = alpha_rho(imgs, a, b)
    lab = classify(alpha, rho, rho_max, eps)

    n_a, n_b = int((lab == CLASS_A).sum()), int((lab == CLASS_B).sum())
    counts = {name: int((lab == i).sum()) for i, name in enumerate(CLASS_NAMES)}
    blur = proportion(counts["floue"], N, conf)
    other = proportion(counts["autre"], N, conf)
    sharp = proportion(n_a + n_b, N, conf)
    ph = p_hat(n_a, n_b, conf)

    # --- MSE (somme des carrés sur les pixels, échelle 0-1) ---
    flat = lambda x: x.reshape(len(x), -1).astype(np.float64)
    y, ya, yb, yr = flat(imgs), flat(a), flat(b), flat(real[sidx])
    sq_real = ((y - yr) ** 2).sum(axis=1)
    sq_a, sq_b = ((y - ya) ** 2).sum(axis=1), ((y - yb) ** 2).sum(axis=1)
    sq_exp = p * sq_a + (1 - p) * sq_b
    floor = bayes_floor(delta[sidx], p)
    fl = float(floor.mean())
    ratio_real = float(sq_real.mean() / fl) if fl > 0 else float("nan")
    ratio_exp = float(sq_exp.mean() / fl) if fl > 0 else float("nan")
    se_real = float((sq_real - floor).std(ddof=1) / np.sqrt(N) / fl) if fl > 0 and N > 1 else float("nan")

    # --- petit objet et artefacts ---
    gx, gy = s_a.gx[sidx], s_a.gy[sidx]
    q = to_uint8(imgs)                      # ce que l'on enregistrerait : détection sur 8 bits
    rec = gem_recall(q, gx, gy, cfg)
    hall = gem_hallucination(q, gx, gy, cfg)
    off = off_palette_fraction(q, cfg)
    off_true = off_palette_fraction(to_uint8(real), cfg)
    blk = block_edge_excess(imgs, token_grid)
    blk_true = block_edge_excess(real, token_grid)

    out = {
        "model": getattr(pred, "name", str(pred)),
        "p": p, "n_states": len(bset), "n_samples": n_samples, "n": N, "rho_max": rho_max, "eps": eps,
        "counts": counts,
        "blur": blur.as_dict(), "other": other.as_dict(), "sharp": sharp.as_dict(),
        "p_hat": ph.as_dict(), "p_abs_err": abs(ph.value - p) if ph.n else float("nan"),
        "p_hat_covers_p": bool(ph.lo <= p <= ph.hi) if ph.n else False,
        "alpha_mean": float(np.nanmean(alpha)),
        "mse": {"sq_err_realized": float(sq_real.mean()), "sq_err_expected": float(sq_exp.mean()),
                "bayes_floor": fl, "ratio_realized": ratio_real, "ratio_expected": ratio_exp,
                "ratio_realized_se": se_real,
                "per_pixel_realized": float(sq_real.mean() / y.shape[1])},
        "recall": proportion(int(rec.sum()), N, conf).as_dict(),
        "hallucination": proportion(int(hall.sum()), N, conf).as_dict(),
        "off_palette": float(off.mean()), "off_palette_truth": float(off_true.mean()),
        "block_edge_excess": float(blk.mean()), "block_edge_excess_truth": float(blk_true.mean()),
        "per_sample": {"label": lab, "alpha": alpha, "rho": rho, "state": sidx,
                       "recall": rec, "sq_err_realized": sq_real},
    }
    if keep_images:
        out["images"] = imgs
    return out


def summarize(res: dict) -> dict:
    """Version légère (JSON) d'un résultat : sans les tableaux par échantillon ni les images."""
    return {k: v for k, v in res.items() if k not in ("per_sample", "images")}


def one_line(res: dict) -> str:
    c = res["counts"]
    n = res["n"]
    ph = res["p_hat"]
    return (f"{res['model']:<34} p={res['p']:.2f}  floue {100*c['floue']/n:5.1f}%  autre {100*c['autre']/n:5.1f}%  "
            f"p̂={ph['value']:.3f} [{ph['ci_lo']:.3f};{ph['ci_hi']:.3f}]  "
            f"MSE/plancher {res['mse']['ratio_expected']:.2f} (tirée {res['mse']['ratio_realized']:.2f})  "
            f"rappel {100*res['recall']['value']:5.1f}%")


# ---------------------------------------------------------------------------------------
# Contrôles et verdicts à une exécution
# ---------------------------------------------------------------------------------------
def check_h0b(res: dict, band: tuple[float, float] = (1.00, 1.10), leak_z: float = 3.0) -> dict:
    """H0b : régression, MSE entre 1,00 et 1,10 fois le plancher de Bayes.

    `fit_ok` : rapport en espérance dans la bande (un rapport nettement au-dessus signale un
    sous-apprentissage). `leak` : rapport contre l'image tirée significativement SOUS 1 (plus de
    `leak_z` erreurs-types), signe d'une information sur la branche dans l'entrée.
    """
    m = res["mse"]
    fit_ok = band[0] <= m["ratio_expected"] <= band[1]
    z = (1.0 - m["ratio_realized"]) / m["ratio_realized_se"] if m["ratio_realized_se"] > 0 else 0.0
    leak = bool(z > leak_z)
    return {"ratio_expected": m["ratio_expected"], "ratio_realized": m["ratio_realized"],
            "leak_z": float(z), "fit_ok": bool(fit_ok), "leak": leak, "pass": bool(fit_ok and not leak)}


def check_blur_below(res: dict, threshold: float = 0.05) -> dict:
    """H0c, H0d : moins de 5 % de flou (borne haute de l'IC comprise pour l'information)."""
    b = res["blur"]
    return {"blur": b["value"], "ci_hi": b["ci_hi"], "threshold": threshold, "pass": bool(b["value"] < threshold)}
