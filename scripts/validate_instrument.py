#!/usr/bin/env python
"""Valide l'instrument de mesure AVANT de mesurer un modèle : règle ρmax et contrôle H0a (sans entraînement).

Exemples
    uv run scripts/validate_instrument.py                         # data/p*/test_bifurcation.npz
    uv run scripts/validate_instrument.py --noise 0.02            # autre niveau de bruit synthétique
    uv run scripts/validate_instrument.py --data data --out reports/instrument.json

ρmax est réglé sur des images pures bruitées par un bruit SYNTHÉTIQUE (σ = 0,01 par défaut) : c'est un choix
de l'instrument, pas une propriété d'un modèle. Le rapport donne aussi la sensibilité à 0,5 x, 1 x, 2 x ρmax.
Sortie : reports/instrument.json (lu par scripts/evaluate.py).
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from flou.eval import instrument


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--noise", type=float, default=0.01, help="écart-type du bruit synthétique (échelle 0-1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("reports/instrument.json"))
    args = ap.parse_args()

    files = sorted(args.data.glob("p*/test_bifurcation.npz"))
    if not files:
        raise SystemExit(f"aucun test_bifurcation.npz dans {args.data}/p*/ : lancer d'abord "
                         "uv run scripts/generate_data.py --split test --p 0.0 0.1 0.3 0.5")
    rep = instrument.validate_instrument(files, noise_sigma=args.noise, seed=args.seed)
    path = instrument.save_report(rep, args.out)

    h = rep["h0a"]
    print(f"{rep['n_pairs']} paires (A, B) sur {len(files)} jeux de test ; δ dans [{rep['delta_range'][0]:.2f}, {rep['delta_range'][1]:.2f}]")
    print(f"ρmax = {rep['rho_max']:.4f}  (bruit synthétique σ = {rep['noise_sigma']}, quantile {rep['quantile']})")
    if rep["avertissement_taille"]:
        print(rep["avertissement_taille"])
    print("H0a (contrôle de l'instrument, sans entraînement) :")
    print(json.dumps({k: h[k] for k in h if not isinstance(h[k], (list, dict))}, indent=1, default=float))
    print("sensibilité au seuil :")
    for k, v in rep["sensitivity"].items():
        print(f"  {k:>5} : ρmax = {v['rho_max']:.4f}  {json.dumps({m: v[m] for m in v if m != 'rho_max'}, default=float)}")
    print(f"rapport : {path}")
    if not h.get("pass", False):
        raise SystemExit("H0a ÉCHOUE : l'instrument n'est pas valide, ne mesurer aucun modèle avant de comprendre pourquoi.")


if __name__ == "__main__":
    main()
