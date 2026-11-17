"""Classifieur contrefactuel (cours préparatoire, chap. 2, et dossier §1.9).

Pour un état de bifurcation, le simulateur rend les deux images possibles du pas suivant, A et B
(distance δ = ||A - B||). Une image prédite ŷ est projetée sur le segment [B, A] :

    α̂ = <ŷ - B, A - B> / δ²  (borné à [0, 1]),     ρ = ||ŷ - (α̂ A + (1 - α̂) B)|| / δ

α̂ dit OÙ se trouve l'image entre B (0) et A (1) ; ρ dit à QUELLE DISTANCE du segment. Avec ε = 0,2 :

    nette-A  si α̂ >= 1 - ε  et ρ <= ρmax        floue  si ε < α̂ < 1 - ε  et ρ <= ρmax
    nette-B  si α̂ <= ε      et ρ <= ρmax        autre  si ρ > ρmax

ρmax se règle sur les mélanges synthétiques (contrôle H0a), jamais sur les sorties d'un modèle.

Limite connue (cours, §2.2) : le mélange idéal a α̂ = p ; il est donc classé « flou » seulement pour
p dans ]0,2 ; 0,8[. Aux p extrêmes l'instrument sous-estime le flou : on rapporte aussi α̂.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 0.2
# étiquettes entières (stables : elles sont écrites dans les fichiers de résultats)
CLASS_A, CLASS_B, CLASS_BLUR, CLASS_OTHER = 0, 1, 2, 3
CLASS_NAMES = ("nette-A", "nette-B", "floue", "autre")


def as_float(img: np.ndarray) -> np.ndarray:
    """uint8 -> [0, 1] ; les flottants sont supposés déjà dans [0, 1]."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64, copy=False)


def alpha_rho(yhat, A, B) -> tuple[np.ndarray, np.ndarray]:
    """(α̂, ρ) pour un lot : yhat de forme (N, ...), A et B de forme (N, ...) ou d'un seul exemple.

    Un exemple isolé (tableau 1D ou (H, W, 3)) renvoie deux scalaires. α̂ est borné à [0, 1] ;
    ρ est normalisé par δ. Si A == B (pas de bifurcation), α̂ et ρ valent nan.
    """
    y = as_float(yhat)
    single = y.ndim in (1, 3)
    if single:
        y = y[None]
    y = y.reshape(y.shape[0], -1)
    a = as_float(A).reshape(-1, y.shape[1])
    b = as_float(B).reshape(-1, y.shape[1])
    d = a - b
    delta2 = (d * d).sum(axis=1)
    safe = np.where(delta2 > 0, delta2, 1.0)
    alpha = np.clip(((y - b) * d).sum(axis=1) / safe, 0.0, 1.0)
    resid = y - (alpha[:, None] * a + (1.0 - alpha[:, None]) * b)
    rho = np.sqrt((resid * resid).sum(axis=1)) / np.sqrt(safe)
    ok = delta2 > 0
    alpha = np.where(ok, alpha, np.nan)
    rho = np.where(ok, rho, np.nan)
    if single:
        return alpha[0], rho[0]
    return alpha, rho


def classify(alpha, rho, rho_max: float, eps: float = EPS) -> np.ndarray:
    """Étiquettes entières (CLASS_*) à partir de (α̂, ρ)."""
    alpha = np.asarray(alpha, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    lab = np.full(alpha.shape, CLASS_BLUR, dtype=np.int8)
    lab[alpha >= 1 - eps] = CLASS_A
    lab[alpha <= eps] = CLASS_B
    lab[(rho > rho_max) | np.isnan(alpha) | np.isnan(rho)] = CLASS_OTHER
    return lab


def classify_images(yhat, A, B, rho_max: float, eps: float = EPS) -> dict:
    alpha, rho = alpha_rho(yhat, A, B)
    return {"alpha": alpha, "rho": rho, "label": classify(alpha, rho, rho_max, eps)}


# ---------------------------------------------------------------------------------------
# Mesure EXPLORATOIRE : le même classifieur restreint au rail de la bifurcation
#
# Déclarée dans DECISIONS.md (révision du 17 novembre 2026) à la suite de l'incident J9. Elle ne
# remplace rien : les verdicts H0 à H5 restent calculés sur l'image entière, par les fonctions
# ci-dessus. Motif : ρ est normalisé par δ, qui ne mesure que le déplacement de l'ennemi, si bien
# qu'un agent décalé d'un pixel suffit à faire basculer une image en « autre » (ρ ≈ 0,39 sur le
# mélange idéal) alors que la bifurcation, elle, est correctement tranchée. Restreindre le calcul
# aux pixels où A et B diffèrent sépare deux questions : « la bifurcation est-elle tranchée,
# mélangée ou ratée ? » et « le reste de l'image est-il juste ? ».
# ---------------------------------------------------------------------------------------
def dilate(mask: np.ndarray, k: int = 2) -> np.ndarray:
    """Dilatation carrée de rayon `k` d'un masque (..., H, W).

    Passe par un remplissage puis des tranches : elle ne reboucle pas d'un bord à l'autre de
    l'image, contrairement à un décalage circulaire.
    """
    mask = np.asarray(mask, dtype=bool)
    if k <= 0:
        return mask
    h, w = mask.shape[-2:]
    pad = np.zeros(mask.shape[:-2] + (h + 2 * k, w + 2 * k), dtype=bool)
    pad[..., k:k + h, k:k + w] = mask
    out = np.zeros_like(mask)
    for dy in range(2 * k + 1):
        for dx in range(2 * k + 1):
            out |= pad[..., dy:dy + h, dx:dx + w]
    return out


def rail_mask(A, B, k: int = 2) -> np.ndarray:
    """Masque (..., H, W) des pixels où A et B diffèrent, dilaté de `k` pixels."""
    return dilate(np.abs(as_float(A) - as_float(B)).sum(-1) > 1e-6, k)


def alpha_rho_rail(yhat, A, B, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(α̂, ρ) calculés sur le seul rail de la bifurcation. Mesure exploratoire, voir ci-dessus.

    δ est inchangé : le rail contient par construction tous les pixels où A et B diffèrent, donc
    ‖A − B‖ restreint au rail vaut ‖A − B‖ entier. Seul le reste ‖ŷ − projection‖ change.
    """
    m = rail_mask(A, B, k)[..., None]
    return alpha_rho(as_float(yhat) * m, as_float(A) * m, as_float(B) * m)


def class_counts(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels)
    return {name: int((labels == i).sum()) for i, name in enumerate(CLASS_NAMES)}


def class_fractions(labels: np.ndarray) -> dict[str, float]:
    n = max(1, int(np.asarray(labels).size))
    return {k: v / n for k, v in class_counts(labels).items()}


def ideal_class(p: float, eps: float = EPS) -> int:
    """Classe du régresseur parfait (mélange idéal pA + (1-p)B, ρ = 0) : α̂ = p."""
    if p >= 1 - eps:
        return CLASS_A
    if p <= eps:
        return CLASS_B
    return CLASS_BLUR


# ---------------------------------------------------------------------------------------
# Validation de l'instrument sans entraînement (H0a)
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Synthetic:
    """Images synthétiques dont la bonne réponse est connue."""
    images: np.ndarray        # (N, ...) float32 dans [0, 1]
    index: np.ndarray         # indice de l'état (A, B) correspondant
    kind: np.ndarray          # 0 = pure A, 1 = pure B, 2 = mélange, 3 = objet détruit (autre)
    alpha: np.ndarray         # α vrai (nan pour « autre »)


KIND_PURE_A, KIND_PURE_B, KIND_MIX, KIND_BROKEN = 0, 1, 2, 3


def make_synthetic(A: np.ndarray, B: np.ndarray, rng: np.random.Generator, noise_sigma: float = 0.0,
                   n_mix: int | None = None, bg_color=(12, 12, 20), gem_xy: np.ndarray | None = None,
                   gem_size: int = 2) -> Synthetic:
    """Pour chaque état : image A pure, image B pure, un mélange αA + (1-α)B avec α ~ U(0,1) et,
    si `gem_xy` est donné, l'image A dont le petit objet est effacé (réponse attendue : « autre »).

    `noise_sigma` ajoute un bruit gaussien par pixel (échelle 0-1) puis quantifie en 8 bits : il
    imite l'imperfection d'un générateur sans introduire de flou systématique.
    """
    A = as_float(A)
    B = as_float(B)
    n = A.shape[0]
    m = n if n_mix is None else n_mix
    idx_mix = rng.integers(0, n, m)
    alpha = rng.random(m)
    shape = (-1,) + (1,) * (A.ndim - 1)
    mix = alpha.reshape(shape) * A[idx_mix] + (1 - alpha.reshape(shape)) * B[idx_mix]
    parts = [A, B, mix]
    index = [np.arange(n), np.arange(n), idx_mix]
    kind = [np.full(n, KIND_PURE_A), np.full(n, KIND_PURE_B), np.full(m, KIND_MIX)]
    alphas = [np.ones(n), np.zeros(n), alpha]
    if gem_xy is not None:
        broken = A.copy()
        bg = np.asarray(bg_color, dtype=np.float64) / 255.0
        for i in range(n):
            gx, gy = int(gem_xy[i, 0]), int(gem_xy[i, 1])
            broken[i, gy:gy + gem_size, gx:gx + gem_size] = bg
        parts.append(broken)
        index.append(np.arange(n))
        kind.append(np.full(n, KIND_BROKEN))
        alphas.append(np.full(n, np.nan))
    imgs = np.concatenate(parts)
    if noise_sigma > 0:
        imgs = np.clip(imgs + rng.normal(0.0, noise_sigma, imgs.shape), 0.0, 1.0)
        imgs = np.round(imgs * 255.0) / 255.0
    return Synthetic(imgs.astype(np.float32), np.concatenate(index), np.concatenate(kind),
                     np.concatenate(alphas))


def _metrics(kind: np.ndarray, alpha_true: np.ndarray, alpha_hat: np.ndarray, rho: np.ndarray, rho_max: float,
             eps: float = EPS) -> dict:
    lab = classify(alpha_hat, rho, rho_max, eps)
    mix = kind == KIND_MIX
    pure_a, pure_b, broken = kind == KIND_PURE_A, kind == KIND_PURE_B, kind == KIND_BROKEN
    out = {
        "rho_max": float(rho_max),
        "alpha_mae_mix": float(np.abs(alpha_hat[mix] - alpha_true[mix]).mean()) if mix.any() else float("nan"),
        "pure_a_ok": float((lab[pure_a] == CLASS_A).mean()) if pure_a.any() else float("nan"),
        "pure_b_ok": float((lab[pure_b] == CLASS_B).mean()) if pure_b.any() else float("nan"),
        "rho_pure_q995": float(np.quantile(rho[pure_a | pure_b], 0.995)) if (pure_a | pure_b).any() else float("nan"),
        "rho_mix_max": float(rho[mix].max()) if mix.any() else float("nan"),
    }
    if (pure_a | pure_b).any():
        out["pure_ok"] = float(((lab[pure_a] == CLASS_A).sum() + (lab[pure_b] == CLASS_B).sum())
                               / (pure_a.sum() + pure_b.sum()))
    if mix.any():
        # les mélanges au sens strict (ρ = 0 hors bruit) ne doivent jamais être « autre »
        out["mix_not_other"] = float((lab[mix] != CLASS_OTHER).mean())
    if broken.any():
        out["broken_flagged_other"] = float((lab[broken] == CLASS_OTHER).mean())
        out["rho_broken_min"] = float(rho[broken].min())
    return out


def evaluate_synthetic(syn: Synthetic, A: np.ndarray, B: np.ndarray, rho_max: float, eps: float = EPS) -> dict:
    """Mesures de H0a sur un jeu synthétique : erreur de α̂ sur les mélanges, taux de bonne
    classification des images pures, taux de détection des images « autre »."""
    a, b = as_float(A)[syn.index], as_float(B)[syn.index]
    alpha_hat, rho = alpha_rho(syn.images, a, b)
    return _metrics(syn.kind, syn.alpha, alpha_hat, rho, rho_max, eps)


CHUNK = 250     # paires par bloc : 250 x 4 images x 12 288 valeurs en float64 ≈ 100 Mo par tableau, quelques copies


def synthetic_stats(A: np.ndarray, B: np.ndarray, rng: np.random.Generator, noise_sigma: float,
                    with_mix: bool = True, gem_xy: np.ndarray | None = None, chunk: int = CHUNK) -> dict:
    """(α̂, ρ, genre, α vrai) de toutes les images synthétiques, calculés BLOC PAR BLOC.

    Fabriquer d'un coup les images synthétiques de 3 000 paires (4 familles d'images, plusieurs copies pour le bruit)
    demande plusieurs Go : on ne garde de chaque bloc que les scalaires. Pour un jeu d'au plus `chunk` paires le
    flux aléatoire est identique à celui d'une fabrication d'un seul bloc."""
    n = len(A)
    out = {"alpha_hat": [], "rho": [], "kind": [], "alpha": []}
    for lo in range(0, n, chunk):
        sl = slice(lo, min(n, lo + chunk))
        syn = make_synthetic(A[sl], B[sl], rng, noise_sigma=noise_sigma, n_mix=None if with_mix else 0,
                             gem_xy=None if gem_xy is None else gem_xy[sl])
        a, b = as_float(A[sl])[syn.index], as_float(B[sl])[syn.index]
        ah, rho = alpha_rho(syn.images, a, b)
        out["alpha_hat"].append(ah)
        out["rho"].append(rho)
        out["kind"].append(syn.kind)
        out["alpha"].append(syn.alpha)
        del syn, a, b
    return {k: np.concatenate(v) for k, v in out.items()}


def tune_rho_max(A: np.ndarray, B: np.ndarray, noise_sigma: float, rng: np.random.Generator,
                 quantile: float = 0.995, margin: float = 1.0) -> float:
    """Plus petit ρmax qui classe correctement `quantile` des images pures bruitées (bruit de
    rendu de niveau `noise_sigma`), multiplié par `margin`. Les mélanges purs ont ρ = 0 et ne
    contraignent pas le seuil."""
    st = synthetic_stats(A, B, rng, noise_sigma, with_mix=False)
    return float(np.quantile(st["rho"], quantile) * margin)


def h0a_check(A: np.ndarray, B: np.ndarray, gem_xy: np.ndarray | None, rho_max: float,
              noise_sigma: float, seed: int = 0, eps: float = EPS) -> dict:
    """Contrôle H0a : sans aucun entraînement, erreur moyenne sur α̂ < 0,05 et plus de 99 % des
    images pures bien classées. Renvoie les mesures et le verdict (`pass`)."""
    rng = np.random.default_rng(seed)
    st = synthetic_stats(A, B, rng, noise_sigma, with_mix=True, gem_xy=gem_xy)
    out = _metrics(st["kind"], st["alpha"], st["alpha_hat"], st["rho"], rho_max, eps)
    out["noise_sigma"] = float(noise_sigma)
    out["pass_alpha"] = bool(out["alpha_mae_mix"] < 0.05)
    out["pass_pure"] = bool(out["pure_ok"] > 0.99)
    out["pass"] = out["pass_alpha"] and out["pass_pure"]
    return out
