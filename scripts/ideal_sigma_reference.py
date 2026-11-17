#!/usr/bin/env python
"""Référence EXPLORATOIRE pour `sigma_sweep.py` : ce que ferait un débruiteur EXACT, un pas depuis du bruit pur.

NumPy seulement (ni PyTorch, ni GPU, ni poids). À lancer AVANT `sigma_sweep.py`, pour que la lecture du balayage
soit fixée avant de voir ses chiffres.

Modèle. L'image suivante vaut A (probabilité p) ou B (probabilité 1 − p). Le débruiteur exact rend la moyenne
a posteriori D(x ; σ) = w·A + (1 − w)·B, avec, pour x = σ·n (le point de départ de l'échantillonneur d'Euler) :

    logit w = ⟨x, A − B⟩ / σ²  −  (‖A‖² − ‖B‖²) / (2σ²)  +  logit p            (loi logistique ; lecture : Bishop, §4.2)

⟨x, A − B⟩/σ² suit N(0, (‖A − B‖/σ)²) : l'écart-type du logit est d/σ. Donc, pour un débruiteur exact :
  - σ ≫ d (σ ≳ 40) : w reste près de p, l'image est un mélange → « floue », ρ = 0 ;
  - σ ≲ d/2 (σ ≲ 7) : le bruit seul suffit à trancher, w tombe vers 0 ou 1 → image NETTE, ρ = 0 aussi.
Le « flou » n'existe donc qu'aux grands σ, et ρ (la classe « autre ») vaut 0 à tout σ. Ce qui sépare un réseau
réel de ce modèle est le grain (ρ élevé) et la dispersion de α̂ plus large que celle prévue à ce σ.

Limites : y est supposé égal à A ou B exactement (le reste de l'image est parfait) ; la sortie du réseau n'est pas
exacte ; le point de départ x = σ·n n'est dans la loi d'entraînement (image + bruit) que si σ ≳ 40 : à σ petit,
l'image de départ a une moyenne que le réseau n'a jamais vue (voir le rapport signal/bruit imprimé).

Exemple
    python scripts/ideal_sigma_reference.py --p 0.3
    python scripts/ideal_sigma_reference.py --p 0.3 --sigmas 1 2 5 10 20 30 40 60 80 --draws 40
"""
import _bootstrap

import argparse
from pathlib import Path

import numpy as np

from flou.data.dataset import BifurcationSet
from flou.eval.classifier import EPS, as_float
from flou.train.runs import test_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1, 2, 5, 10, 20, 30, 40, 60, 80])
    ap.add_argument("--draws", type=int, default=40, help="tirages de bruit par état")
    ap.add_argument("--n-states", type=int, default=300, help="états de test utilisés (0 : tous) ; 300 suffisent, les quantiles ne bougent pas au dixième de point près")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a_img, b_img, _ = BifurcationSet.load(test_file(args.data, args.p)).counterfactuals()
    if args.n_states:
        a_img, b_img = a_img[:args.n_states], b_img[:args.n_states]
    A, B = (as_float(x) * 2.0 - 1.0 for x in (a_img, b_img))          # échelle [-1, 1], celle du modèle
    n = len(A)
    diff = (A - B).reshape(n, -1)
    d = np.linalg.norm(diff, axis=1)
    norm_gap = (A.reshape(n, -1) ** 2).sum(1) - (B.reshape(n, -1) ** 2).sum(1)
    rms, mean = float(np.sqrt((A ** 2).mean())), float(A.mean())
    rng = np.random.default_rng(args.seed)
    logit_p = np.log(args.p / (1.0 - args.p))

    print(f"p = {args.p}, {n} états ; d = ‖A − B‖ (échelle [-1, 1]) : médiane {np.median(d):.1f}")
    print(f"images cibles : moyenne par pixel {mean:+.3f}, RMS {rms:.3f}")
    print("Débruiteur EXACT, un pas depuis du bruit pur (poids a posteriori w de A ; flou : 0,2 < w < 0,8)")
    print(f"{'sigma':>6} {'floue':>8} {'nettes':>8} {'autre':>7} {'w p5':>7} {'w p95':>7} "
          f"{'sd logit = d/sigma':>19} {'signal/bruit':>13}")
    for s in args.sigmas:
        w = []
        for _ in range(args.draws):
            x = s * rng.standard_normal(diff.shape)
            logit = (x * diff).sum(1) / s ** 2 - norm_gap / (2 * s ** 2) + logit_p
            w.append(1.0 / (1.0 + np.exp(-logit)))
        w = np.concatenate(w)
        floue = float(np.mean((w > EPS) & (w < 1 - EPS)))
        print(f"{s:>6g} {floue:>7.1%} {1 - floue:>8.1%} {0.0:>6.0%} {np.quantile(w, 0.05):>7.3f} "
              f"{np.quantile(w, 0.95):>7.3f} {np.median(d) / s:>19.2f} {rms / s:>13.3f}")
    print("« autre » = 0 par construction : toute valeur mesurée au-dessus vient du réseau (grain, point de départ).")


if __name__ == "__main__":
    main()
