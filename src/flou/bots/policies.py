"""Les quatre bots qui produisent les données.

Ils lisent l'état exact (ce sont des scripts, pas des agents appris) et sont vectorisés :
`act(state, rng)` renvoie une action par partie. Ils n'influencent pas l'ennemi, qui est
autonome : la fréquence des états de bifurcation ne dépend donc pas du bot.

    random        actions uniformes
    greedy        va vers le petit objet (axe le plus éloigné d'abord, contourne les blocages)
    expert        va vers le petit objet en évitant l'ennemi (anticipe les deux branches)
    noisy_expert  expert, mais action aléatoire avec probabilité epsilon
"""
from __future__ import annotations

import numpy as np

from ..game.config import DOWN, LEFT, N_ACTIONS, NOOP, RIGHT, UP, GameConfig
from ..game.dynamics import agent_step, box_gap, enemy_step, enemy_xy
from ..game.state import State


class Bot:
    name = "bot"

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg

    def reset(self, n_envs: int) -> None:  # les bots sont sans mémoire ; point d'extension
        return None

    def act(self, s: State, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class RandomBot(Bot):
    name = "random"

    def act(self, s, rng):
        return rng.integers(0, N_ACTIONS, s.n).astype(np.int32)


class GreedyBot(Bot):
    """Avance vers le petit objet. Si l'axe principal est bloqué (pilier, obstacle), il essaie
    l'autre axe, puis contourne par un pas perpendiculaire aléatoire. `jitter` : probabilité
    d'une action aléatoire, qui évite les blocages durables."""

    name = "greedy"

    def __init__(self, cfg, jitter: float = 0.05):
        super().__init__(cfg)
        self.jitter = jitter

    def act(self, s, rng):
        cfg = self.cfg
        A, G = cfg.agent_size, cfg.gem_size
        ddx = (2 * s.gx + G) - (2 * s.ax + A)       # écart de centres, en demi-pixels
        ddy = (2 * s.gy + G) - (2 * s.ay + A)
        tol = A + G                                  # il y a ramassage si |écart| < A + G
        need_x, need_y = np.abs(ddx) >= tol, np.abs(ddy) >= tol
        act_x = np.where(ddx > 0, RIGHT, LEFT)
        act_y = np.where(ddy > 0, DOWN, UP)
        prefer_x = np.abs(ddx) >= np.abs(ddy)
        c1, n1 = np.where(prefer_x, act_x, act_y), np.where(prefer_x, need_x, need_y)
        c2, n2 = np.where(prefer_x, act_y, act_x), np.where(prefer_x, need_y, need_x)
        b1 = agent_step(s, c1, cfg)[2]
        b2 = agent_step(s, c2, cfg)[2]
        coin = rng.random(s.n) < 0.5
        perp = np.where(prefer_x, np.where(coin, UP, DOWN), np.where(coin, LEFT, RIGHT))
        act = np.where(n1 & ~b1, c1, np.where(n2 & ~b2, c2, np.where(n1, perp, NOOP)))
        wild = rng.random(s.n) < self.jitter
        return np.where(wild, rng.integers(1, N_ACTIONS, s.n), act).astype(np.int32)


class ExpertBot(Bot):
    """Choisit l'action qui minimise (distance au petit objet) + (risque lié à l'ennemi).

    Le risque est évalué sur un horizon de `horizon` pas : on suppose que l'agent répète
    l'action candidate, et on compare sa position à celles de l'ennemi pour CHAQUE branche
    possible (le pire cas compte). Un horizon d'un seul pas ne suffit pas : l'ennemi va trois
    fois plus vite que l'agent et balaie toute la bande centrale. Jusqu'à 5 pas, il y a au
    plus une bifurcation à venir, donc « toujours A » et « toujours B » couvrent tous les cas.
    """

    name = "expert"

    def __init__(self, cfg, horizon: int = 3, margin: int = 4, hit_penalty: float = 100.0,
                 margin_weight: float = 4.0):
        super().__init__(cfg)
        if not 1 <= horizon <= 5:
            raise ValueError("horizon doit être dans [1, 5]")
        self.horizon, self.margin = horizon, margin
        self.hit_penalty, self.margin_weight = hit_penalty, margin_weight

    def _enemy_futures(self, s: State):
        """Positions de l'ennemi aux pas 1..horizon, pour la branche A puis la branche B."""
        out = []
        for branch in (0, 1):
            eo, ev = s.eo, s.ev
            traj = []
            for _ in range(self.horizon):
                eo, ev = enemy_step(eo, ev, np.full(s.n, branch, dtype=np.int32), self.cfg)
                traj.append(enemy_xy(eo, self.cfg))
            out.append(traj)
        return out

    def act(self, s, rng):
        cfg = self.cfg
        A, G, E = cfg.agent_size, cfg.gem_size, cfg.enemy_size
        futures = self._enemy_futures(s)
        scores = np.empty((s.n, N_ACTIONS))
        for a in range(N_ACTIONS):
            act = np.full(s.n, a, dtype=np.int32)
            path = s.copy()
            risk = np.zeros(s.n)
            for h in range(self.horizon):
                path.ax, path.ay, _ = agent_step(path, act, cfg)   # l'agent répète l'action
                for traj in futures:                                # pire cas sur les branches
                    ex, ey = traj[h]
                    gap = box_gap(path.ax, path.ay, A, ex, ey, E)
                    r = np.where(gap < 0, self.hit_penalty,
                                 np.clip(self.margin - gap, 0, None) * self.margin_weight)
                    risk = np.maximum(risk, r / (h + 1))
                if h == 0:
                    ax1, ay1 = path.ax.copy(), path.ay.copy()      # position après le 1er pas
            ddx = (2 * s.gx + G) - (2 * ax1 + A)
            ddy = (2 * s.gy + G) - (2 * ay1 + A)
            dist = np.hypot(ddx, ddy) / 2.0                         # pixels
            scores[:, a] = -dist - risk - (0.01 if a == NOOP else 0.0) + rng.random(s.n) * 1e-3
        return np.argmax(scores, axis=1).astype(np.int32)


class NoisyExpertBot(ExpertBot):
    name = "noisy_expert"

    def __init__(self, cfg, epsilon: float = 0.2, **kw):
        super().__init__(cfg, **kw)
        self.epsilon = epsilon

    def act(self, s, rng):
        good = super().act(s, rng)
        wild = rng.random(s.n) < self.epsilon
        return np.where(wild, rng.integers(0, N_ACTIONS, s.n), good).astype(np.int32)


BOTS = {c.name: c for c in (RandomBot, GreedyBot, ExpertBot, NoisyExpertBot)}


def make_bot(name: str, cfg: GameConfig, **kwargs) -> Bot:
    if name not in BOTS:
        raise KeyError(f"bot inconnu {name!r} ; choix : {sorted(BOTS)}")
    return BOTS[name](cfg, **kwargs)
