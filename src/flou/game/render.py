"""Rendu : fonction pure de l'état. Aucune information cachée n'entre dans l'image.

Ordre de superposition (du fond vers le dessus) : murs, obstacle, ennemi, agent, petit objet.
Le petit objet est dessiné en dernier : il reste visible même sous l'ennemi ou l'obstacle,
ce qui rend sa détection (rappel) non ambiguë.
"""
from __future__ import annotations

import numpy as np

from .config import GameConfig
from .dynamics import advance, enemy_xy, geometry
from .state import State


def _draw_boxes(img: np.ndarray, x: np.ndarray, y: np.ndarray, size: int, color) -> None:
    b = np.arange(img.shape[0])[:, None, None]
    r = np.arange(size)
    ys = (y[:, None] + r)[:, :, None]
    xs = (x[:, None] + r)[:, None, :]
    img[b, ys, xs] = np.asarray(color, dtype=np.uint8)


def render(s: State, cfg: GameConfig) -> np.ndarray:
    """Images (B, 64, 64, 3) uint8 pour un lot d'états."""
    g = geometry(cfg)
    img = np.broadcast_to(g.base, (s.n, cfg.size, cfg.size, 3)).copy()
    ex, ey = enemy_xy(s.eo, cfg)
    _draw_boxes(img, s.ox, s.oy, cfg.obstacle_size, cfg.color_obstacle)
    _draw_boxes(img, ex, ey, cfg.enemy_size, cfg.color_enemy)
    _draw_boxes(img, s.ax, s.ay, cfg.agent_size, cfg.color_agent)
    _draw_boxes(img, s.gx, s.gy, cfg.gem_size, cfg.color_gem)
    return img


def render_counterfactual(s: State, action, cfg: GameConfig):
    """Les deux images possibles du pas suivant pour un même état et une même action.

    Renvoie (img_A, img_B, state_A, state_B). Aux états de bifurcation, A est la branche
    de probabilité p (l'ennemi part à droite) et B l'autre ; ailleurs, A == B.
    """
    action = np.asarray(action, dtype=np.int32)
    ones = np.ones(s.n, dtype=np.int32)
    s_a, _ = advance(s, action, ones, cfg)
    s_b, _ = advance(s, action, 0 * ones, cfg)
    return render(s_a, cfg), render(s_b, cfg), s_a, s_b


def branch_delta(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """δ = ||A - B|| (norme euclidienne sur les pixels, échelle 0-255 ramenée à 0-1)."""
    d = (img_a.astype(np.float64) - img_b.astype(np.float64)) / 255.0
    return np.sqrt((d ** 2).reshape(d.shape[0], -1).sum(axis=1))


def render_one(s: State, cfg: GameConfig) -> np.ndarray:
    """Image (64, 64, 3) du premier état d'un lot (pratique en interactif)."""
    return render(s[:1], cfg)[0]
