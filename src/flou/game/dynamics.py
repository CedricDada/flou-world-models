"""Dynamique du jeu : fonctions pures, vectorisées sur un lot de parties.

    advance(state, action, branch, cfg) -> (state_suivant, événements)

est la SEULE fonction qui fait évoluer le monde. La branche de l'ennemi est un argument :
- dans une partie normale, `VecMiniGame` la tire (A avec probabilité p) ;
- pour le rendu contrefactuel, on appelle `advance` deux fois avec branch = A puis B.

Ordre de mise à jour d'un pas (voir docs/GAME_SPEC.md) :
  1. agent (action) ; 2. obstacle (rebonds) ; 3. ennemi (branche si au centre du rail) ;
  4. événements, ramassage du petit objet.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import BRANCH_A, NO_BRANCH, NOOP, GameConfig
from .state import Events, State

_I32 = np.int32
# déplacement unitaire par action : noop, haut, bas, gauche, droite
DX = np.array([0, 0, 0, -1, 1], dtype=_I32)
DY = np.array([0, -1, 1, 0, 0], dtype=_I32)


# ---------------------------------------------------------------------------------------
# Géométrie précalculée (murs, image de fond, emplacements du petit objet)
# ---------------------------------------------------------------------------------------
class Geometry:
    def __init__(self, cfg: GameConfig):
        n = cfg.size
        mask = np.zeros((n, n), dtype=bool)
        w = cfg.wall
        mask[:w, :] = mask[-w:, :] = mask[:, :w] = mask[:, -w:] = True
        for x, y, bw, bh in cfg.pillars:
            mask[y:y + bh, x:x + bw] = True
        self.size = n
        self.wall_mask = mask
        # table des sommes cumulées : nombre de pixels de mur dans n'importe quelle boîte en O(1)
        sat = np.zeros((n + 1, n + 1), dtype=_I32)
        sat[1:, 1:] = mask.cumsum(0).cumsum(1)
        self.sat = sat
        self.gem_spots = np.array(cfg.gem_spots, dtype=_I32)  # (K, 2)
        base = np.empty((n, n, 3), dtype=np.uint8)
        base[:] = cfg.color_bg
        base[mask] = cfg.color_wall
        self.base = base

    def hits_wall(self, x: np.ndarray, y: np.ndarray, size: int) -> np.ndarray:
        """Vrai si la boîte size x size de coin (x, y) touche un mur ou sort de l'arène."""
        n = self.size
        x2, y2 = x + size, y + size
        outside = (x < 0) | (y < 0) | (x2 > n) | (y2 > n)
        xc, yc = np.clip(x, 0, n), np.clip(y, 0, n)
        x2c, y2c = np.clip(x2, 0, n), np.clip(y2, 0, n)
        s = self.sat
        count = s[y2c, x2c] - s[yc, x2c] - s[y2c, xc] + s[yc, xc]
        return outside | (count > 0)

    def farthest_spot(self, ax: np.ndarray, ay: np.ndarray, a_size: int, g_size: int):
        """Emplacement du petit objet le plus éloigné de l'agent (égalité : plus petit indice)."""
        sp = self.gem_spots
        # coordonnées de centres doublées pour rester en entiers
        dx = 2 * sp[None, :, 0] + g_size - 2 * ax[:, None] - a_size
        dy = 2 * sp[None, :, 1] + g_size - 2 * ay[:, None] - a_size
        k = np.argmax(dx * dx + dy * dy, axis=1)
        return sp[k, 0], sp[k, 1]


@lru_cache(maxsize=16)
def geometry(cfg: GameConfig) -> Geometry:
    return Geometry(cfg)


# ---------------------------------------------------------------------------------------
# Petits outils
# ---------------------------------------------------------------------------------------
def boxes_overlap(x1, y1, s1, x2, y2, s2) -> np.ndarray:
    """Chevauchement strict de deux boîtes carrées (le simple contact ne compte pas)."""
    return (x1 < x2 + s2) & (x2 < x1 + s1) & (y1 < y2 + s2) & (y2 < y1 + s1)


def box_gap(x1, y1, s1, x2, y2, s2) -> np.ndarray:
    """Écart entre deux boîtes (négatif si elles se chevauchent, 0 au contact)."""
    gx = np.maximum(x1 - (x2 + s2), x2 - (x1 + s1))
    gy = np.maximum(y1 - (y2 + s2), y2 - (y1 + s1))
    return np.maximum(gx, gy)


def enemy_xy(eo: np.ndarray, cfg: GameConfig) -> tuple[np.ndarray, np.ndarray]:
    """Coin haut-gauche de l'ennemi pour un décalage de rail eo."""
    hx, hy = cfg.hub
    return hx + eo * cfg.enemy_step, np.full_like(eo, hy)


def is_branch_state(s: State) -> np.ndarray:
    """Vrai si l'ennemi est au centre du rail : la prochaine transition est une bifurcation."""
    return s.eo == 0


# ---------------------------------------------------------------------------------------
# Les trois morceaux de la dynamique
# ---------------------------------------------------------------------------------------
def agent_step(s: State, action: np.ndarray, cfg: GameConfig):
    """Déplacement de l'agent. Renvoie (ax, ay, bloqué). L'agent est arrêté par les murs et
    par l'obstacle (seulement si le mouvement CRÉE un chevauchement : il peut en sortir)."""
    g = geometry(cfg)
    a = np.asarray(action, dtype=_I32)
    A, O = cfg.agent_size, cfg.obstacle_size
    nx = s.ax + DX[a] * cfg.agent_speed
    ny = s.ay + DY[a] * cfg.agent_speed
    moving = a != NOOP
    by_wall = g.hits_wall(nx, ny, A)
    by_obstacle = boxes_overlap(nx, ny, A, s.ox, s.oy, O) & ~boxes_overlap(s.ax, s.ay, A, s.ox, s.oy, O)
    blocked = moving & (by_wall | by_obstacle)
    go = moving & ~blocked
    return np.where(go, nx, s.ax), np.where(go, ny, s.ay), blocked


def obstacle_step(s: State, cfg: GameConfig):
    """Obstacle : mouvement rectiligne, rebond sur les murs (axe par axe). Déterministe."""
    g = geometry(cfg)
    O = cfg.obstacle_size
    tx = s.ox + s.ovx
    bx = g.hits_wall(tx, s.oy, O)
    ox = np.where(bx, s.ox, tx)
    ovx = np.where(bx, -s.ovx, s.ovx)
    ty = s.oy + s.ovy
    by = g.hits_wall(ox, ty, O)
    oy = np.where(by, s.oy, ty)
    ovy = np.where(by, -s.ovy, s.ovy)
    return ox, oy, ovx, ovy, bx, by


def enemy_step(eo: np.ndarray, ev: np.ndarray, branch: np.ndarray, cfg: GameConfig):
    """Ennemi sur son rail : aller-retour sur un bras, bifurcation au centre.

    Au centre (eo == 0) la branche décide du bras : A -> droite (+1), B -> gauche (-1).
    Ailleurs, il continue ; au bout du bras (|eo| == spoke_len) il fait demi-tour.
    """
    eo = np.asarray(eo, dtype=_I32)
    hub = eo == 0
    ev = np.where(hub, np.where(branch == BRANCH_A, 1, -1), ev).astype(_I32)
    eo_new = eo + ev
    at_end = np.abs(eo_new) == cfg.spoke_len
    ev_new = np.where(at_end, -np.sign(eo_new), ev).astype(_I32)
    return eo_new, ev_new


# ---------------------------------------------------------------------------------------
# La fonction centrale
# ---------------------------------------------------------------------------------------
def advance(s: State, action, branch, cfg: GameConfig) -> tuple[State, Events]:
    """Un pas de jeu. `branch` : tableau (B,) de -1, 0 ou 1 ; il DOIT valoir 0 ou 1 là où
    l'ennemi est au centre du rail (ailleurs il est ignoré)."""
    g = geometry(cfg)
    a = np.asarray(action, dtype=_I32)
    br = np.asarray(branch, dtype=_I32)
    if a.shape != (s.n,) or br.shape != (s.n,):
        raise ValueError(f"action et branch doivent avoir la forme ({s.n},)")
    if np.any(is_branch_state(s) & (br == NO_BRANCH)):
        raise ValueError("branche manquante dans un état de bifurcation")
    A, O, E, G = cfg.agent_size, cfg.obstacle_size, cfg.enemy_size, cfg.gem_size

    ax, ay, blocked = agent_step(s, a, cfg)
    ox, oy, ovx, ovy, bounce_x, bounce_y = obstacle_step(s, cfg)
    eo, ev = enemy_step(s.eo, s.ev, br, cfg)
    ex, ey = enemy_xy(eo, cfg)

    hit_enemy = boxes_overlap(ax, ay, A, ex, ey, E)
    hit_obstacle = boxes_overlap(ax, ay, A, ox, oy, O)
    pickup = boxes_overlap(ax, ay, A, s.gx, s.gy, G)
    ngx, ngy = g.farthest_spot(ax, ay, A, G)
    gx = np.where(pickup, ngx, s.gx)
    gy = np.where(pickup, ngy, s.gy)

    nxt = State(ax, ay, ox, oy, ovx, ovy, eo, ev, gx, gy)
    used_branch = np.where(is_branch_state(s), br, NO_BRANCH).astype(np.int8)
    ev_ = Events(blocked, hit_enemy, hit_obstacle, bounce_x, bounce_y, pickup, used_branch)
    return nxt, ev_


# ---------------------------------------------------------------------------------------
# États initiaux
# ---------------------------------------------------------------------------------------
def _resample_until_ok(rng, n, lo, hi_x, hi_y, bad_fn, max_iter=200):
    x = rng.integers(lo, hi_x + 1, n).astype(_I32)
    y = rng.integers(lo, hi_y + 1, n).astype(_I32)
    for _ in range(max_iter):
        bad = bad_fn(x, y)
        if not bad.any():
            return x, y
        k = int(bad.sum())
        x[bad] = rng.integers(lo, hi_x + 1, k)
        y[bad] = rng.integers(lo, hi_y + 1, k)
    raise RuntimeError("impossible de placer les objets : arène trop encombrée ?")


def initial_state(cfg: GameConfig, n: int, rng: np.random.Generator) -> State:
    """n états initiaux : obstacle, ennemi, petit objet et agent tirés au hasard, sans
    chevauchement entre l'agent et les autres objets. Tous sont des états atteignables."""
    g = geometry(cfg)
    A, O, E, G, S = cfg.agent_size, cfg.obstacle_size, cfg.enemy_size, cfg.gem_size, cfg.size
    w = cfg.wall
    # obstacle : position libre, vitesses (±1 ou ±2, jamais nulles)
    ox, oy = _resample_until_ok(rng, n, w, S - w - O, S - w - O, lambda x, y: g.hits_wall(x, y, O))
    speeds = np.array(cfg.obstacle_speeds, dtype=_I32)
    ovx = rng.choice(speeds, n) * rng.choice(np.array([-1, 1], dtype=_I32), n)
    ovy = rng.choice(speeds, n) * rng.choice(np.array([-1, 1], dtype=_I32), n)
    # ennemi : tout décalage de rail atteignable ; au bout d'un bras le sens est imposé
    L = cfg.spoke_len
    eo = rng.integers(-L, L + 1, n).astype(_I32)
    ev = rng.choice(np.array([-1, 1], dtype=_I32), n)
    ev = np.where(np.abs(eo) == L, -np.sign(eo), ev).astype(_I32)
    ex, ey = enemy_xy(eo, cfg)
    # petit objet : un des emplacements
    k = rng.integers(0, len(g.gem_spots), n)
    gx, gy = g.gem_spots[k, 0].copy(), g.gem_spots[k, 1].copy()
    # agent : libre, hors des autres objets
    def bad(x, y):
        return (g.hits_wall(x, y, A) | boxes_overlap(x, y, A, ox, oy, O)
                | boxes_overlap(x, y, A, ex, ey, E) | boxes_overlap(x, y, A, gx, gy, G))
    ax, ay = _resample_until_ok(rng, n, w, S - w - A, S - w - A, bad)
    return State(ax, ay, ox, oy, ovx, ovy, eo, ev, gx, gy)
