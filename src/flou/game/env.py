"""Environnements : version vectorisée (données, expériences) et version Gymnasium (une partie).

`VecMiniGame` fait avancer `n_envs` parties en parallèle avec NumPy. Les épisodes sont
synchronisés (même longueur `max_steps`, pas de fin prématurée) : après `max_steps` pas,
`truncated` vaut True pour toutes les parties et on rappelle `reset()`.
"""
from __future__ import annotations

import numpy as np

from .config import NO_BRANCH, GameConfig
from .dynamics import advance, initial_state, is_branch_state
from .render import render
from .state import State

try:  # Gymnasium est facultatif : le cœur du projet n'en dépend pas
    import gymnasium as gym
    from gymnasium import spaces

    _BASE = gym.Env
except ImportError:  # pragma: no cover - dépend de l'installation
    gym = None
    spaces = None
    _BASE = object


class VecMiniGame:
    def __init__(self, cfg: GameConfig, n_envs: int, seed: int | None = None, render_obs: bool = True):
        self.cfg = cfg
        self.n = int(n_envs)
        self.render_obs = render_obs
        self.rng = np.random.default_rng(seed)
        self.state: State | None = None
        self.t = 0
        self.score = np.zeros(self.n, dtype=np.int32)

    # ---------------------------------------------------------------------------------
    def reset(self, seed: int | None = None, state: State | None = None):
        """Nouvelle série de parties. `state` impose les états initiaux (tests, contrefactuels)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = state.copy() if state is not None else initial_state(self.cfg, self.n, self.rng)
        if self.state.n != self.n:
            raise ValueError(f"état initial de taille {self.state.n}, attendu {self.n}")
        self.t = 0
        self.score[:] = 0
        return self._obs(), self._info(None, None)

    def step(self, actions, branch=None):
        """Un pas. `branch` (facultatif) force la branche aux états de bifurcation ; sinon elle
        est tirée : A avec probabilité p. Renvoie (obs, reward, terminated, truncated, info)."""
        if self.state is None:
            raise RuntimeError("appeler reset() d'abord")
        if self.t >= self.cfg.max_steps:
            raise RuntimeError("épisode terminé : appeler reset()")
        hub = is_branch_state(self.state)
        u = self.rng.random(self.n)                       # toujours consommé : flux reproductible
        drawn = np.where(hub, (u < self.cfg.p).astype(np.int32), NO_BRANCH)
        if branch is not None:
            drawn = np.where(hub, np.asarray(branch, dtype=np.int32), NO_BRANCH)
        self.state, ev = advance(self.state, actions, drawn, self.cfg)
        self.t += 1
        reward = ev.pickup.astype(np.float32) - ev.hit_enemy.astype(np.float32)
        self.score += reward.astype(np.int32)
        terminated = np.zeros(self.n, dtype=bool)
        truncated = np.full(self.n, self.t >= self.cfg.max_steps)
        return self._obs(), reward, terminated, truncated, self._info(ev, drawn)

    # ---------------------------------------------------------------------------------
    def _obs(self):
        return render(self.state, self.cfg) if self.render_obs else None

    def _info(self, ev, branch):
        info = {"state": self.state.copy(), "t": self.t, "score": self.score.copy()}
        if ev is not None:
            info["events"] = ev
            info["branch"] = ev.branch
        return info


class MiniGameEnv(_BASE):
    """Une seule partie, interface Gymnasium : reset(seed=, options=) -> (obs, info) ;
    step(action) -> (obs, reward, terminated, truncated, info).

    `options={"state": vecteur de 10 entiers}` impose l'état initial. L'état exact et les
    drapeaux d'événements sont dans `info` ("state" : vecteur de 10 entiers ; "branch" :
    -1, 0 ou 1 ; "events" : dictionnaire de booléens).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, cfg: GameConfig | None = None, seed: int | None = None, render_mode: str | None = "rgb_array"):
        self.cfg = cfg or GameConfig()
        self.render_mode = render_mode
        self._vec = VecMiniGame(self.cfg, 1, seed=seed)
        self._last_obs = None
        if spaces is not None:
            self.observation_space = spaces.Box(0, 255, (self.cfg.size, self.cfg.size, 3), dtype=np.uint8)
            self.action_space = spaces.Discrete(5)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            super().reset(seed=seed)
        init = None
        if options and "state" in options:
            init = State.from_array(np.asarray(options["state"]))
        obs, info = self._vec.reset(seed=seed, state=init)
        self._last_obs = obs[0]
        return self._last_obs, self._single(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._vec.step(np.array([int(action)]))
        self._last_obs = obs[0]
        return self._last_obs, float(reward[0]), bool(terminated[0]), bool(truncated[0]), self._single(info)

    def render(self):
        return self._last_obs

    @staticmethod
    def _single(info: dict) -> dict:
        out = {"state": info["state"].to_array()[0], "t": info["t"], "score": int(info["score"][0])}
        if "events" in info:
            out["events"] = {k: v[0].item() for k, v in info["events"].to_dict().items()}
            out["branch"] = int(info["branch"][0])
        return out
