"""Mathématiques d'EDM (Karras et al. 2022) en NumPy : calendrier de σ, préconditionnement, perte,
et échantillonneur d'Euler avec le débruiteur EXACT d'un cas jouet à deux issues.

Les fonctions `precond_coefficients` et `loss_weight` n'emploient que des opérateurs arithmétiques :
elles s'appliquent donc aussi bien à des tableaux NumPy qu'à des tenseurs PyTorch (`diffusion.py` les
réutilise telles quelles). Les identités d'EDM sont testées ici, sans PyTorch.

Cas jouet (cours, §5.7) : deux issues A = 1 (probabilité p) et B = 0, débruiteur exact
    D(x, σ) = sigmoid( logit(p) + (x - 1/2) / σ² ).
"""
from __future__ import annotations

import numpy as np

SIGMA_DATA = 0.5
SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7.0
P_MEAN, P_STD = -1.2, 1.2          # σ = exp(P_MEAN + P_STD · N(0,1)) à l'entraînement


def karras_sigmas(n: int, sigma_max: float = SIGMA_MAX, sigma_min: float = SIGMA_MIN,
                  rho: float = RHO) -> np.ndarray:
    """Calendrier décroissant de n niveaux suivi de 0 (donc n pas d'Euler). Pour n = 1 : [σmax, 0],
    l'unique pas de débruitage part de σmax et arrive à D(x; σmax) (la moyenne conditionnelle)."""
    if n < 1:
        raise ValueError("n >= 1")
    if n == 1:
        return np.array([sigma_max, 0.0])
    i = np.arange(n)
    s = (sigma_max ** (1 / rho) + i / (n - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return np.append(s, 0.0)


def precond_coefficients(sigma, sigma_data: float = SIGMA_DATA):
    """(c_skip, c_out, c_in) tels que D(x; σ) = c_skip x + c_out F(c_in x, ln σ / 4)."""
    s2, d2 = sigma ** 2, sigma_data ** 2
    return d2 / (s2 + d2), sigma * sigma_data / (s2 + d2) ** 0.5, 1.0 / (s2 + d2) ** 0.5


def loss_weight(sigma, sigma_data: float = SIGMA_DATA):
    """λ(σ) = (σ² + σ_d²) / (σ σ_d)² = 1 / c_out² : rend l'erreur de F de variance 1 pour tout σ."""
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


def sample_sigma(rng: np.random.Generator, n: int, p_mean: float = P_MEAN, p_std: float = P_STD) -> np.ndarray:
    return np.exp(p_mean + p_std * rng.standard_normal(n))


# ---------------------------------------------------------------------------------------
# Cas jouet à deux issues
# ---------------------------------------------------------------------------------------
def _sigmoid(u):
    return 1.0 / (1.0 + np.exp(-np.clip(u, -700, 700)))


def exact_denoiser(x, sigma, p: float):
    """Espérance de y sachant x = y + σ ξ, pour y = 1 (probabilité p) ou 0."""
    return _sigmoid(np.log(p / (1 - p)) + (x - 0.5) / sigma ** 2)


def euler_sample_toy(p: float, n: int, n_samples: int = 400_000, sigma_max: float = SIGMA_MAX,
                     sigma_min: float = SIGMA_MIN, rho: float = RHO, seed: int = 1) -> np.ndarray:
    """Échantillons du cas jouet par l'échantillonneur d'Euler à n pas avec le débruiteur exact :
    x ← x + (σ' − σ)(x − D(x; σ))/σ. Reproduit la simulation du cours (§5.7)."""
    rng = np.random.default_rng(seed)
    sg = karras_sigmas(n, sigma_max, sigma_min, rho)
    x = rng.standard_normal(n_samples) * sg[0]
    for s, sn in zip(sg[:-1], sg[1:]):
        x = x + (sn - s) * (x - exact_denoiser(x, s, p)) / s
    return x


def toy_p_hat(x: np.ndarray, eps: float = 0.2) -> tuple[float, float]:
    """(p̂, part de flou) d'après le classifieur du chap. 2 : α̂ = clip(x, 0, 1)."""
    a = np.clip(x, 0.0, 1.0)
    n_a, n_b = int((a >= 1 - eps).sum()), int((a <= eps).sum())
    blur = float(((a > eps) & (a < 1 - eps)).mean())
    return (n_a / (n_a + n_b) if n_a + n_b else float("nan")), blur
