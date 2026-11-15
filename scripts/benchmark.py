#!/usr/bin/env python
"""Mesure le débit du simulateur, par bot, avec et sans rendu des images.

    uv run scripts/benchmark.py                 # 256 parties en parallèle
    uv run scripts/benchmark.py --n-envs 1024

Objectif du dossier : 200 000 images par bot en quelques minutes sur le CPU de Colab.
"""
import _bootstrap  # noqa: F401

import argparse
import platform
import time

import numpy as np

from flou.bots import BOTS, make_bot
from flou.game import GameConfig, VecMiniGame


def bench(cfg, bot_name, n_envs, render_obs, episodes=2):
    env = VecMiniGame(cfg, n_envs, seed=0, render_obs=render_obs)
    bot = make_bot(bot_name, cfg)
    rng = np.random.default_rng(1)
    t0 = time.perf_counter()
    steps = 0
    for _ in range(episodes):
        env.reset()
        for _ in range(cfg.max_steps):
            env.step(bot.act(env.state, rng))
            steps += n_envs
    return steps / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--target", type=int, default=200_000, help="images visées par bot")
    args = ap.parse_args()
    cfg = GameConfig()
    print(f"machine : {platform.processor() or platform.machine()}, Python {platform.python_version()}, "
          f"NumPy {np.__version__}, {args.n_envs} parties en parallèle\n")
    print(f"{'bot':<14}{'états/s':>12}{'états+images/s':>18}{'200 000 images':>20}")
    for name in BOTS:
        fast = bench(cfg, name, args.n_envs, render_obs=False)
        full = bench(cfg, name, args.n_envs, render_obs=True)
        print(f"{name:<14}{fast:>12,.0f}{full:>18,.0f}{args.target / full:>17.1f} s")


if __name__ == "__main__":
    main()
