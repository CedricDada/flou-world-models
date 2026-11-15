"""Collecte de transitions avec les bots.

On stocke des ÉTATS (entiers, 10 par image) et non des images : le rendu est une fonction
pure de l'état, rapide (voir scripts/benchmark.py), et 200 000 images de 64x64x3 pèseraient
2,4 Go contre environ 4 Mo pour les états. Les images sont rendues à la volée à la lecture.

Les épisodes ont tous la même longueur T = cfg.max_steps, d'où des tableaux rectangulaires :
    states   (n_épisodes, T+1, 10) int16   états s_0 ... s_T
    actions  (n_épisodes, T)       int8    a_t, appliquée en s_t
    branch   (n_épisodes, T)       int8    branche tirée à la transition t -> t+1 (-1 : aucune)
    + drapeaux d'événements (n_épisodes, T) bool : pickup, hit_enemy, hit_obstacle,
      blocked, bounce_x, bounce_y
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..bots import make_bot
from ..game import GameConfig, VecMiniGame

SPLIT_ID = {"train": 0, "val": 1, "test": 2}
FLAG_NAMES = ("pickup", "hit_enemy", "hit_obstacle", "blocked", "bounce_x", "bounce_y")
FORMAT_VERSION = 1


def stream_seeds(seed: int, split: str, bot_index: int = 0) -> tuple[int, int]:
    """Deux graines (partie, bot) indépendantes et distinctes pour chaque (seed, split, bot).
    Les trois jeux (train, val, test) ne partagent donc aucun flux aléatoire."""
    ss = np.random.SeedSequence([seed, SPLIT_ID[split], bot_index])
    a, b = ss.spawn(2)
    return int(a.generate_state(1)[0]), int(b.generate_state(1)[0])


def collect_episodes(cfg: GameConfig, bot_name: str, n_episodes: int, env_seed: int, bot_seed: int,
                     n_envs: int = 256, bot_kwargs: dict | None = None) -> dict[str, np.ndarray]:
    T = cfg.max_steps
    env = VecMiniGame(cfg, n_envs, seed=env_seed, render_obs=False)
    bot = make_bot(bot_name, cfg, **(bot_kwargs or {}))
    brng = np.random.default_rng(bot_seed)
    chunks: dict[str, list] = {k: [] for k in ("states", "actions", "branch", *FLAG_NAMES)}
    done = 0
    while done < n_episodes:
        env.reset()
        bot.reset(n_envs)
        S = np.empty((n_envs, T + 1, 10), dtype=np.int16)
        Act = np.empty((n_envs, T), dtype=np.int8)
        Br = np.empty((n_envs, T), dtype=np.int8)
        Fl = {k: np.empty((n_envs, T), dtype=bool) for k in FLAG_NAMES}
        S[:, 0] = env.state.to_array()
        for t in range(T):
            a = bot.act(env.state, brng)
            _, _, _, _, info = env.step(a)
            S[:, t + 1] = info["state"].to_array()
            Act[:, t] = a
            Br[:, t] = info["branch"]
            for k in FLAG_NAMES:
                Fl[k][:, t] = getattr(info["events"], k)
        for k, v in (("states", S), ("actions", Act), ("branch", Br), *Fl.items()):
            chunks[k].append(v)
        done += n_envs
    return {k: np.concatenate(v)[:n_episodes] for k, v in chunks.items()}


def save_npz(path: str | Path, data: dict[str, np.ndarray], meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data, meta=np.array(json.dumps(meta)))
    return path


def load_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        data = {k: z[k] for k in z.files if k != "meta"}
    return data, meta


def generate_split(cfg: GameConfig, split: str, bots: list[str], n_transitions: int, seed: int,
                   out_dir: str | Path, n_envs: int = 256, bot_kwargs: dict | None = None) -> list[Path]:
    """Un fichier par bot ; n_transitions est réparti à parts égales entre les bots."""
    T = cfg.max_steps
    per_bot = -(-n_transitions // len(bots))             # partie entière supérieure
    n_episodes = -(-per_bot // T)
    paths = []
    for i, bot in enumerate(bots):
        env_seed, bot_seed = stream_seeds(seed, split, i)
        data = collect_episodes(cfg, bot, n_episodes, env_seed, bot_seed, n_envs, (bot_kwargs or {}).get(bot))
        meta = {"format": FORMAT_VERSION, "config": cfg.to_dict(), "bot": bot, "split": split,
                "seed": seed, "n_envs": n_envs, "n_episodes": n_episodes, "episode_len": T,
                "bot_kwargs": (bot_kwargs or {}).get(bot, {})}
        fname = f"{split}_{bot}.npz"
        paths.append(save_npz(Path(out_dir) / f"p{cfg.p:.2f}" / fname, data, meta))
    return paths
