#!/usr/bin/env python
"""Images de démonstration : une partie par bot, et le rendu contrefactuel A / B / moyenne.

    uv run scripts/demo.py --out docs/img      (sans uv : python scripts/demo.py --out docs/img)
Nécessite Pillow (groupe viz de uv, installé par défaut ; ou : pip install pillow).
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np

from flou.bots import BOTS, make_bot
from flou.game import GameConfig, State, VecMiniGame, is_branch_state, render, render_counterfactual
from flou.game.viz import contact_sheet


def play(cfg, bot_name, steps, seed):
    env = VecMiniGame(cfg, 1, seed=seed)
    bot = make_bot(bot_name, cfg)
    rng = np.random.default_rng(seed + 1)
    obs, info = env.reset()
    frames, states = [obs[0]], [info["state"]]
    for _ in range(steps):
        obs, *_rest, info = env.step(bot.act(env.state, rng))
        frames.append(obs[0])
        states.append(info["state"])
    return frames, states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/img"))
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--steps", type=int, default=11)
    args = ap.parse_args()
    cfg = GameConfig().with_p(args.p)

    # 1) une partie par bot, 12 images consécutives (tous les 2 pas pour voir du mouvement)
    rows, labels = [], []
    for name in BOTS:
        frames, _ = play(cfg, name, args.steps * 2, args.seed)
        rows.append(frames[::2][: args.steps + 1])
        labels.append([f"{name} t={2 * i}" for i in range(len(rows[-1]))])
    out = contact_sheet(rows, args.out / "parties_par_bot.png", scale=3, labels=labels)
    print("écrit", out)

    # 2) rendu contrefactuel : pour 4 états de bifurcation, [état | A | B | moyenne]
    env = VecMiniGame(cfg, 8, seed=args.seed, render_obs=False)
    bot = make_bot("greedy", cfg)
    rng = np.random.default_rng(args.seed)
    _, info = env.reset()
    states = [info["state"]]
    for _ in range(60):
        states.append(env.step(bot.act(env.state, rng))[-1]["state"])
    st = State.concatenate(states)
    hubs = np.nonzero(is_branch_state(st))[0]
    sel = st[hubs[np.linspace(0, len(hubs) - 1, 4).astype(int)]]
    img_now = render(sel, cfg)
    action = np.zeros(sel.n, dtype=np.int32)
    img_a, img_b, _, _ = render_counterfactual(sel, action, cfg)
    mean = ((img_a.astype(np.float32) + img_b.astype(np.float32)) / 2).round().astype(np.uint8)
    mix = (cfg.p * img_a.astype(np.float32) + (1 - cfg.p) * img_b.astype(np.float32)).round().astype(np.uint8)
    rows = [[img_now[i], img_a[i], img_b[i], mix[i]] for i in range(sel.n)]
    labels = [["etat t", "A (droite)", "B (gauche)", f"p*A+(1-p)*B, p={cfg.p}"] for _ in rows]
    out = contact_sheet(rows, args.out / "contrefactuel.png", scale=4, labels=labels)
    print("écrit", out)


if __name__ == "__main__":
    main()
