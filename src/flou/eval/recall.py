"""Rappel du petit objet (H3a) et artefacts d'image (taxonomie du dossier, §1.9 : (c) et (e)).

Le petit objet de 2 x 2 pixels est dessiné en dernier (au-dessus de tout) : il est toujours
entièrement visible dans une image réelle, donc sa détection n'est jamais ambiguë.
Le rappel est la probabilité qu'il soit présent AU BON ENDROIT dans l'image prédite ; l'endroit est
déterministe (l'agent, l'obstacle et le petit objet ne dépendent pas de la branche de l'ennemi).
"""
from __future__ import annotations

import numpy as np

from ..game.config import GameConfig
from .classifier import as_float
from .probe import color_mask
from .stats import Proportion, proportion

GEM_TOL = 0.25   # écart max (échelle 0-1, par canal) à la couleur du petit objet
GEM_MIN_PIXELS = 2  # sur 4 pixels : « présent » si au moins la moitié est reconnaissable


def gem_pixels(imgs: np.ndarray, cfg: GameConfig, tol: float = GEM_TOL) -> np.ndarray:
    """Masque (N, H, W) des pixels à la couleur du petit objet."""
    col = np.asarray(cfg.color_gem, dtype=np.float64) / 255.0
    return color_mask(imgs, col, tol)


def _box_mask(gx, gy, size: int, h: int, w: int) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    gx = np.asarray(gx)[:, None, None]
    gy = np.asarray(gy)[:, None, None]
    return (xs >= gx) & (xs < gx + size) & (ys >= gy) & (ys < gy + size)


def gem_recall(imgs: np.ndarray, gx, gy, cfg: GameConfig, tol: float = GEM_TOL,
               min_pixels: int = GEM_MIN_PIXELS) -> np.ndarray:
    """Booléens (N,) : le petit objet est reconnaissable à l'emplacement attendu (gx, gy)."""
    m = gem_pixels(imgs, cfg, tol)
    box = _box_mask(gx, gy, cfg.gem_size, m.shape[1], m.shape[2])
    return (m & box).sum(axis=(1, 2)) >= min_pixels


def gem_hallucination(imgs: np.ndarray, gx, gy, cfg: GameConfig, tol: float = GEM_TOL,
                      min_pixels: int = GEM_MIN_PIXELS) -> np.ndarray:
    """Booléens (N,) : au moins `min_pixels` pixels de la couleur du petit objet HORS de l'emplacement
    attendu (objet dupliqué, déplacé ou inventé)."""
    m = gem_pixels(imgs, cfg, tol)
    box = _box_mask(gx, gy, cfg.gem_size, m.shape[1], m.shape[2])
    return (m & ~box).sum(axis=(1, 2)) >= min_pixels


def recall_proportion(imgs, gx, gy, cfg: GameConfig, conf: float = 0.95, **kw) -> Proportion:
    ok = gem_recall(imgs, gx, gy, cfg, **kw)
    return proportion(int(ok.sum()), len(ok), conf)


# ---------------------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------------------
def palette(cfg: GameConfig) -> np.ndarray:
    """Couleurs légitimes (6, 3), échelle 0-1 : fond, mur, obstacle, ennemi, agent, petit objet."""
    cols = (cfg.color_bg, cfg.color_wall, cfg.color_obstacle, cfg.color_enemy, cfg.color_agent, cfg.color_gem)
    return np.asarray(cols, dtype=np.float64) / 255.0


def off_palette_fraction(imgs: np.ndarray, cfg: GameConfig, tol: float = 0.1) -> np.ndarray:
    """Part (N,) des pixels éloignés de plus de `tol` de TOUTES les couleurs légitimes.
    Un mélange flou ou un bruit de décodeur augmente cette part ; une image réelle vaut 0."""
    x = np.asarray(imgs)
    pal = palette(cfg)
    near = np.zeros(x.shape[:3], dtype=bool)
    for c in pal:                                   # une couleur à la fois : pas de tableau (N,H,W,P,3)
        near |= color_mask(x, c, tol)
    return 1.0 - near.mean(axis=(1, 2))


def block_edge_excess(imgs: np.ndarray, g: int = 8) -> np.ndarray:
    """Discontinuité moyenne sur les frontières de blocs g x g moins celle des autres frontières
    de pixels (N,). À comparer à la même mesure sur la vérité terrain : l'excès propre aux jetons
    est la différence entre prédiction et vérité."""
    x = as_float(imgs)
    h, w = x.shape[1:3]
    dx = np.abs(np.diff(x, axis=2)).mean(axis=(1, 3))        # (N, W-1) : frontière entre col j et j+1
    dy = np.abs(np.diff(x, axis=1)).mean(axis=(2, 3))        # (N, H-1)
    ex = ((np.arange(w - 1) + 1) % g) == 0
    ey = ((np.arange(h - 1) + 1) % g) == 0
    edge = np.concatenate([dx[:, ex], dy[:, ey]], axis=1).mean(axis=1)
    inner = np.concatenate([dx[:, ~ex], dy[:, ~ey]], axis=1).mean(axis=1)
    return edge - inner
