"""Sonde analytique : position de chaque objet par sa couleur (aucun apprentissage).

Le rendu est en aplats de couleur sans bruit : le centre de gravité des pixels d'une couleur
donne la position exacte de l'objet, tant qu'il n'est pas partiellement masqué (ordre de
superposition : obstacle, ennemi, agent, petit objet ; voir `flou.game.render`). Cette sonde sert
de vérité de référence pour valider la sonde apprise (CNN, `flou.models.probe`) : une sonde
apprise doit atteindre une RMSE sous 0,5 pixel (cours, chap. 7), et la sonde analytique donne la
RMSE « de plancher » due aux seules occultations.

Sur une image floue ou bruitée, la sonde ne trouve rien (nan) : c'est voulu, c'est une information.
"""
from __future__ import annotations

import numpy as np

from ..game.config import GameConfig
from ..game.dynamics import enemy_xy
from ..game.state import State
from .classifier import as_float

ENTITIES = ("agent", "obstacle", "enemy", "gem")


def _colors(cfg: GameConfig) -> dict[str, np.ndarray]:
    return {
        "agent": np.asarray(cfg.color_agent, dtype=np.float64) / 255.0,
        "obstacle": np.asarray(cfg.color_obstacle, dtype=np.float64) / 255.0,
        "enemy": np.asarray(cfg.color_enemy, dtype=np.float64) / 255.0,
        "gem": np.asarray(cfg.color_gem, dtype=np.float64) / 255.0,
    }


def color_mask(imgs: np.ndarray, color: np.ndarray, tol: float) -> np.ndarray:
    """Pixels dont les trois canaux sont à moins de `tol` (échelle 0-1) de `color`. (N, H, W).
    Les images uint8 prennent un chemin entier (plus rapide, sans conversion en flottants). Les trois
    canaux sont comparés séparément : une réduction `max(axis=-1)` sur un axe de taille 3 est ~10x
    plus lente."""
    imgs = np.asarray(imgs)
    if imgs.dtype == np.uint8:
        c = np.round(np.asarray(color, dtype=np.float64) * 255.0).astype(np.int16)
        lim = tol * 255.0 + 1e-9
        x = imgs.astype(np.int16)
    else:
        c = np.asarray(color, dtype=np.float64)
        lim = tol
        x = as_float(imgs)
    m = np.abs(x[..., 0] - c[0]) <= lim
    m &= np.abs(x[..., 1] - c[1]) <= lim
    m &= np.abs(x[..., 2] - c[2]) <= lim
    return m


class AnalyticProbe:
    """imgs (N, H, W, 3) -> positions (N, 4, 2) = (x, y) du centre de chaque objet, dans l'ordre
    de `ENTITIES`, en coordonnées de centres de pixels ; nan si l'objet n'est pas trouvé."""

    def __init__(self, cfg: GameConfig, tol: float = 0.12, min_pixels: int = 1):
        self.cfg, self.tol, self.min_pixels = cfg, tol, min_pixels
        self.colors = _colors(cfg)

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        imgs = np.asarray(imgs)
        n, h, w, _ = imgs.shape
        ys, xs = np.mgrid[0:h, 0:w]
        out = np.full((n, len(ENTITIES), 2), np.nan)
        for j, name in enumerate(ENTITIES):
            m = color_mask(imgs, self.colors[name], self.tol)
            cnt = m.sum(axis=(1, 2))
            ok = cnt >= self.min_pixels
            safe = np.where(ok, cnt, 1)
            out[:, j, 0] = np.where(ok, (m * xs).sum(axis=(1, 2)) / safe, np.nan)
            out[:, j, 1] = np.where(ok, (m * ys).sum(axis=(1, 2)) / safe, np.nan)
        return out


def true_positions(s: State, cfg: GameConfig) -> np.ndarray:
    """Positions exactes (N, 4, 2) dans le même repère que la sonde : centre de la boîte,
    en coordonnées de centres de pixels (coin + (taille - 1) / 2)."""
    ex, ey = enemy_xy(s.eo, cfg)
    def c(x, y, size):
        h = (size - 1) / 2.0
        return np.stack([np.asarray(x, dtype=np.float64) + h, np.asarray(y, dtype=np.float64) + h], axis=-1)
    return np.stack([c(s.ax, s.ay, cfg.agent_size), c(s.ox, s.oy, cfg.obstacle_size),
                     c(ex, ey, cfg.enemy_size), c(s.gx, s.gy, cfg.gem_size)], axis=1)


def position_errors(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Distance euclidienne (N, 4) en pixels ; nan si l'objet n'a pas été trouvé."""
    return np.linalg.norm(pred - truth, axis=-1)


def probe_report(pred: np.ndarray, truth: np.ndarray) -> dict:
    """RMSE (pixels), part d'objets trouvés et R² par objet. Critère du cours : RMSE < 0,5 pixel."""
    err = position_errors(pred, truth)
    out = {}
    for j, name in enumerate(ENTITIES):
        e = err[:, j]
        found = ~np.isnan(e)
        p, t = pred[found, j], truth[found, j]
        ss_res = float(((p - t) ** 2).sum())
        ss_tot = float(((t - t.mean(axis=0)) ** 2).sum())
        out[name] = {
            "found": float(found.mean()),
            "rmse": float(np.sqrt((e[found] ** 2).mean())) if found.any() else float("nan"),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 and found.any() else float("nan"),
            "max_err": float(e[found].max()) if found.any() else float("nan"),
        }
    return out
