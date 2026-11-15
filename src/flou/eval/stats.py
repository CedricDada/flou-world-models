"""Statistiques de décision du projet (cours préparatoire, chap. 7).

Deux sources de bruit, deux familles d'outils :

* bruit d'**échantillonnage** (quelques milliers d'images tirées d'un même réseau) :
  intervalles binomiaux sur une proportion (`wilson_ci`, `clopper_pearson`, `required_n`) ;
* bruit d'**entraînement** (3 graines par condition) : règles à 3 graines (`unpaired_threshold`,
  `mean_ci_halfwidth`, `paired_threshold`, `bonferroni_*`, `power_unpaired`), `compare_groups`.

Tout est calculé avec scipy ; aucun nombre n'est recopié du cours : les tests les recalculent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as st


# ---------------------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------------------
def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Intervalle exact (Clopper-Pearson) pour k succès sur n essais."""
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= k <= n:
        raise ValueError("k doit être dans [0, n]")
    a = 1.0 - conf
    lo = 0.0 if k == 0 else st.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else st.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Intervalle de Wilson (approximation courante, meilleure que Wald aux petits effectifs)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = st.norm.ppf(1 - (1 - conf) / 2)
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def wald_half_width(p: float, n: int, conf: float = 0.95) -> float:
    """Demi-intervalle normal 1,96·sqrt(p(1-p)/N) (tableau du chap. 7)."""
    return float(st.norm.ppf(1 - (1 - conf) / 2) * math.sqrt(p * (1 - p) / n))


def required_n(p: float, half_width: float, conf: float = 0.95) -> int:
    """Plus petit N tel que le demi-intervalle normal soit <= half_width."""
    z = st.norm.ppf(1 - (1 - conf) / 2)
    return int(math.ceil((z / half_width) ** 2 * p * (1 - p)))


def prob_error_exceeds(p: float, n: int, tol: float) -> float:
    """Pr(|p̂ - p| > tol) pour un modèle parfaitement calibré (approximation normale)."""
    sd = math.sqrt(p * (1 - p) / n)
    return float(2 * st.norm.sf(tol / sd))


@dataclass(frozen=True)
class Proportion:
    k: int
    n: int
    lo: float
    hi: float

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else float("nan")

    def as_dict(self) -> dict:
        return {"k": self.k, "n": self.n, "value": self.value, "ci_lo": self.lo, "ci_hi": self.hi}


def proportion(k: int, n: int, conf: float = 0.95) -> Proportion:
    lo, hi = clopper_pearson(k, n, conf)
    return Proportion(int(k), int(n), lo, hi)


def p_hat(n_a: int, n_b: int, conf: float = 0.95) -> Proportion:
    """p̂ = N_A / (N_A + N_B) sur les seules images nettes (cours, §7.2)."""
    return proportion(n_a, n_a + n_b, conf)


# ---------------------------------------------------------------------------------------
# Règles de décision à peu de graines
# ---------------------------------------------------------------------------------------
def t_crit(df: int, alpha: float = 0.05, k_tests: int = 1) -> float:
    """Quantile bilatéral de Student, avec correction de Bonferroni pour k_tests tests."""
    return float(st.t.ppf(1 - alpha / (2 * k_tests), df))


def unpaired_threshold(n1: int = 3, n2: int = 3, alpha: float = 0.05, k_tests: int = 1) -> float:
    """Seuil, en écarts-types communs s, pour une différence de deux moyennes non appariées.
    3 contre 3 : t_4 · sqrt(2/3) ≈ 2,27 s."""
    return t_crit(n1 + n2 - 2, alpha, k_tests) * math.sqrt(1 / n1 + 1 / n2)


def mean_ci_halfwidth(n: int = 3, alpha: float = 0.05) -> float:
    """Demi-largeur de l'IC d'une moyenne de n graines, en écarts-types s : t_{n-1}/sqrt(n) ≈ 2,48 s."""
    return t_crit(n - 1, alpha) / math.sqrt(n)


def paired_threshold(n: int = 3, alpha: float = 0.05, k_tests: int = 1) -> float:
    """Seuil de la moyenne des n différences, en écarts-types s_d des différences."""
    return t_crit(n - 1, alpha, k_tests) / math.sqrt(n)


def family_wise_error(k: int, alpha: float = 0.05) -> float:
    """Probabilité d'au moins un faux positif parmi k tests indépendants à niveau alpha."""
    return 1.0 - (1.0 - alpha) ** k


def power_unpaired(d: float, n1: int = 3, n2: int = 3, alpha: float = 0.05, k_tests: int = 1) -> float:
    """Puissance du test t non apparié pour un écart de d écarts-types (t non centré)."""
    df = n1 + n2 - 2
    crit = t_crit(df, alpha, k_tests)
    ncp = d / math.sqrt(1 / n1 + 1 / n2)
    return float(st.nct.sf(crit, df, ncp) + st.nct.cdf(-crit, df, ncp))


@dataclass(frozen=True)
class Comparison:
    """Résultat d'une comparaison entre deux conditions, à peu de graines."""
    mean_a: float
    mean_b: float
    diff: float            # mean_a - mean_b
    sd: float              # écart-type commun (non apparié) ou des différences (apparié)
    threshold: float       # seuil en unités de s
    ratio: float           # |diff| / sd : à comparer à `threshold`
    t: float
    p_value: float
    paired: bool
    n_a: int
    n_b: int
    significant: bool      # |diff| > threshold * sd, au niveau (Bonferroni) demandé
    verdict: str           # « concluant » ou « non concluant »

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def compare_groups(a, b, paired: bool = False, alpha: float = 0.05, k_tests: int = 1) -> Comparison:
    """Compare deux séries de mesures (une valeur par graine).

    Non apparié : écart-type commun s, seuil t_{na+nb-2}·sqrt(1/na+1/nb) (2,27 s pour 3 contre 3).
    Apparié (mêmes réseaux dans les deux conditions) : écart-type s_d des différences,
    seuil t_{n-1}/sqrt(n) (2,48 s_d pour n = 3).
    Un résultat sous le seuil est « non concluant », jamais « pas d'effet » (puissance de ~16 %
    pour un écart de 1 s à 3 contre 3).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if paired:
        if a.shape != b.shape:
            raise ValueError("comparaison appariée : séries de même longueur")
        d = a - b
        n = len(d)
        if n < 2:
            raise ValueError("au moins 2 paires")
        sd = float(d.std(ddof=1))
        thr = paired_threshold(n, alpha, k_tests)
        diff = float(d.mean())
        tstat = diff / (sd / math.sqrt(n)) if sd > 0 else math.copysign(math.inf, diff) if diff else 0.0
        p = float(2 * st.t.sf(abs(tstat), n - 1)) if math.isfinite(tstat) else (0.0 if diff else 1.0)
    else:
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            raise ValueError("au moins 2 mesures par condition")
        sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
        sd = float(math.sqrt(sp2))
        thr = unpaired_threshold(na, nb, alpha, k_tests)
        diff = float(a.mean() - b.mean())
        se = sd * math.sqrt(1 / na + 1 / nb)
        tstat = diff / se if se > 0 else math.copysign(math.inf, diff) if diff else 0.0
        p = float(2 * st.t.sf(abs(tstat), na + nb - 2)) if math.isfinite(tstat) else (0.0 if diff else 1.0)
    ratio = abs(diff) / sd if sd > 0 else (math.inf if diff else 0.0)
    sig = bool(ratio > thr)
    return Comparison(float(a.mean()), float(b.mean()), diff, sd, thr, ratio, float(tstat), p, paired,
                      len(a), len(b), sig, "concluant" if sig else "non concluant")


def mean_ci(x, alpha: float = 0.05) -> tuple[float, float, float]:
    """(moyenne, borne basse, borne haute) de l'IC de Student d'une moyenne sur les graines."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    h = t_crit(n - 1, alpha) * float(x.std(ddof=1)) / math.sqrt(n)
    return m, m - h, m + h


# ---------------------------------------------------------------------------------------
# Accord entre relecteurs
# ---------------------------------------------------------------------------------------
def cohen_kappa(r1, r2) -> float:
    """κ de Cohen pour deux relecteurs (étiquettes catégorielles) : (p_o - p_e)/(1 - p_e).

    Si p_e = 1 (les deux relecteurs n'emploient qu'une seule et même catégorie), κ est indéfini : nan.
    """
    r1, r2 = np.asarray(r1), np.asarray(r2)
    if r1.shape != r2.shape or r1.ndim != 1 or len(r1) == 0:
        raise ValueError("deux vecteurs non vides de même longueur")
    cats = np.union1d(r1, r2)
    po = float(np.mean(r1 == r2))
    pe = float(sum(np.mean(r1 == c) * np.mean(r2 == c) for c in cats))
    if math.isclose(pe, 1.0):
        return float("nan")
    return (po - pe) / (1 - pe)


def mean_pairwise_kappa(ratings: np.ndarray) -> float:
    """κ de Cohen moyen sur toutes les paires de relecteurs ; ratings : (relecteurs, éléments)."""
    ratings = np.asarray(ratings)
    r = ratings.shape[0]
    ks = [cohen_kappa(ratings[i], ratings[j]) for i in range(r) for j in range(i + 1, r)]
    ks = [k for k in ks if not math.isnan(k)]
    return float(np.mean(ks)) if ks else float("nan")
