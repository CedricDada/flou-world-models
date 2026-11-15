#!/usr/bin/env python
"""Génère les jeux de données (états, actions, branches) avec les bots.

Exemples
    # 100 000 transitions d'entraînement (bots aléatoire + glouton), p = 0,3
    uv run scripts/generate_data.py --split train --p 0.3 --n-transitions 100000

    # validation (épisodes distincts : autre flux aléatoire)
    uv run scripts/generate_data.py --split val --p 0.3 --n-transitions 20000

    # test : 1 000 états de bifurcation figés, communs à tous les modèles
    uv run scripts/generate_data.py --split test --p 0.3 --n-bifurcations 1000

    # les trois valeurs de p du plan d'expériences d'un coup
    uv run scripts/generate_data.py --split train --p 0.1 0.3 0.5
"""
import _bootstrap  # noqa: F401  (ajoute src/ au chemin)

import argparse
import time
from pathlib import Path

from flou.bots import BOTS
from flou.data import build_bifurcation_set, generate_split
from flou.game import GameConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--p", type=float, nargs="+", default=[0.3], help="probabilité(s) de la branche A")
    ap.add_argument("--bots", nargs="+", default=["random", "greedy"], choices=sorted(BOTS))
    ap.add_argument("--n-transitions", type=int, default=100_000, help="total, réparti entre les bots")
    ap.add_argument("--n-bifurcations", type=int, default=1000, help="taille du jeu de test")
    ap.add_argument("--context", type=int, default=4, help="images de contexte du jeu de test")
    ap.add_argument("--horizon", type=int, default=16, help="actions futures stockées (rollouts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=256, help="parties simulées en parallèle")
    ap.add_argument("--config", type=str, default=None, help="JSON de GameConfig (facultatif)")
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    base = GameConfig.from_json(args.config) if args.config else GameConfig()
    for p in args.p:
        cfg = base.with_p(p)
        t0 = time.time()
        if args.split == "test":
            bs = build_bifurcation_set(cfg, n=args.n_bifurcations, k=args.context, seed=args.seed,
                                       bots=tuple(args.bots), n_envs=args.n_envs, horizon=args.horizon)
            meta = {"config": cfg.to_dict(), "split": "test", "seed": args.seed, "bots": args.bots,
                    "context": args.context, "horizon": args.horizon, "n": len(bs)}
            paths = [bs.save(args.out / f"p{p:.2f}" / "test_bifurcation.npz", meta)]
        else:
            paths = generate_split(cfg, args.split, args.bots, args.n_transitions, args.seed,
                                   args.out, n_envs=args.n_envs)
        for path in paths:
            print(f"p={p:.2f}  {path}  ({path.stat().st_size / 1e6:.1f} Mo, {time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
