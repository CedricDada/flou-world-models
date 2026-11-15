"""Lecture des transitions : fenêtres de k images passées + action -> image suivante.

Aucune dépendance à PyTorch : les fonctions renvoient des tableaux NumPy. Les images sont
rendues à la demande à partir des états (rendu pur), donc rien d'énorme n'est stocké.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..game import GameConfig, State, branch_delta, render, render_counterfactual
from .generate import FLAG_NAMES, collect_episodes, load_npz, save_npz, stream_seeds


class TransitionData:
    """Un ou plusieurs fichiers de transitions (même configuration de jeu).

    Une transition d'indice i est (épisode e, pas t) avec e = i // T et t = i % T :
        contexte  = images des états s_{t-k+1} ... s_t   (répétition de s_0 avant le début)
        action    = a_t
        cible     = image de s_{t+1}
        branch    = branche tirée à cette transition (-1 : pas une bifurcation)
    """

    def __init__(self, paths: list[str | Path], k: int = 4):
        if not paths:
            raise ValueError("aucun fichier")
        parts, metas = zip(*(load_npz(p) for p in paths))
        cfg_dicts = {str(m["config"]) for m in metas}
        if len(cfg_dicts) != 1:
            raise ValueError("les fichiers n'ont pas la même configuration de jeu")
        self.meta = metas[0]
        self.cfg = GameConfig.from_dict(metas[0]["config"])
        self.k = int(k)
        self.T = self.cfg.max_steps
        self.states = np.concatenate([d["states"] for d in parts])
        self.actions = np.concatenate([d["actions"] for d in parts])
        self.branch = np.concatenate([d["branch"] for d in parts])
        self.flags = {f: np.concatenate([d[f] for d in parts]) for f in FLAG_NAMES}
        self.n_episodes = self.states.shape[0]

    def __len__(self) -> int:
        return self.n_episodes * self.T

    # ---------------------------------------------------------------------------------
    def locate(self, idx) -> tuple[np.ndarray, np.ndarray]:
        idx = np.atleast_1d(np.asarray(idx))
        return idx // self.T, idx % self.T

    def is_bifurcation(self) -> np.ndarray:
        """Masque (len,) des transitions dont l'état de départ est un état de bifurcation."""
        return (self.branch >= 0).reshape(-1)

    def bifurcation_share(self) -> float:
        return float(self.is_bifurcation().mean())

    def context_states(self, idx) -> np.ndarray:
        """États des k images de contexte : (n, k, 10) int16."""
        e, t = self.locate(idx)
        steps = np.clip(t[:, None] + np.arange(-self.k + 1, 1)[None, :], 0, None)
        return self.states[e[:, None], steps]

    def get(self, idx, with_images: bool = True) -> dict[str, np.ndarray]:
        e, t = self.locate(idx)
        ctx = self.context_states(idx)
        out = {
            "context_states": ctx,
            "target_state": self.states[e, t + 1],
            "action": self.actions[e, t].astype(np.int64),
            "branch": self.branch[e, t].astype(np.int64),
            "episode": e, "t": t,
        }
        if with_images:
            n, k = ctx.shape[:2]
            out["context"] = render(State.from_array(ctx.reshape(n * k, -1)), self.cfg).reshape(
                n, k, *render_shape(self.cfg))
            out["target"] = render(State.from_array(out["target_state"]), self.cfg)
        return out

    def small_object_pixels(self) -> int:
        return self.cfg.gem_size ** 2


def render_shape(cfg: GameConfig) -> tuple[int, int, int]:
    return cfg.size, cfg.size, 3


# ---------------------------------------------------------------------------------------
# Jeu de test : uniquement des états de bifurcation
# ---------------------------------------------------------------------------------------
class BifurcationSet:
    """États de bifurcation figés, identiques pour tous les modèles (cours
    préparatoire, chap. 9 : « test sur uniquement des états de bifurcation, graine distincte,
    1 000 par valeur de p »).

    Chaque élément donne le contexte (k images), l'action et les DEUX images contrefactuelles
    A et B : c'est ce qui permet de classer une prédiction en nette-A, nette-B, floue ou autre.
    """

    def __init__(self, cfg: GameConfig, ctx_states: np.ndarray, actions: np.ndarray, branch: np.ndarray,
                 future_actions: np.ndarray | None = None):
        self.cfg, self.ctx_states, self.actions, self.branch = cfg, ctx_states, actions, branch
        # actions du bot aux pas t, t+1, ..., t+H-1 (colonne 0 = `actions`) : sert aux rollouts
        self.future_actions = future_actions

    @property
    def horizon(self) -> int:
        return 0 if self.future_actions is None else self.future_actions.shape[1]

    def subset(self, idx) -> "BifurcationSet":
        fa = None if self.future_actions is None else self.future_actions[idx]
        return BifurcationSet(self.cfg, self.ctx_states[idx], self.actions[idx], self.branch[idx], fa)

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def k(self) -> int:
        return self.ctx_states.shape[1]

    def contexts(self) -> np.ndarray:
        n, k = self.ctx_states.shape[:2]
        imgs = render(State.from_array(self.ctx_states.reshape(n * k, -1)), self.cfg)
        return imgs.reshape(n, k, *render_shape(self.cfg))

    def last_state(self) -> State:
        """État courant (instant t) de chaque exemple : dernière image du contexte."""
        return State.from_array(self.ctx_states[:, -1])

    def counterfactuals(self):
        """(image A, image B, δ) pour chaque état : A = l'ennemi part à droite (probabilité p)."""
        img_a, img_b, _, _ = render_counterfactual(self.last_state(), self.actions, self.cfg)
        return img_a, img_b, branch_delta(img_a, img_b)

    def next_states(self) -> tuple[State, State]:
        """États suivants (A, B). L'agent, l'obstacle et le petit objet y sont identiques : seul
        l'ennemi dépend de la branche."""
        _, _, s_a, s_b = render_counterfactual(self.last_state(), self.actions, self.cfg)
        return s_a, s_b

    def realized_images(self) -> np.ndarray:
        """Image réellement suivie (celle de la branche stockée dans `branch`)."""
        img_a, img_b, _ = self.counterfactuals()
        return np.where((self.branch == 1)[:, None, None, None], img_a, img_b)

    def save(self, path: str | Path, meta: dict) -> Path:
        data = {"ctx_states": self.ctx_states, "actions": self.actions, "branch": self.branch}
        if self.future_actions is not None:
            data["future_actions"] = self.future_actions
        return save_npz(path, data, meta)

    @classmethod
    def load(cls, path: str | Path) -> "BifurcationSet":
        data, meta = load_npz(path)
        return cls(GameConfig.from_dict(meta["config"]), data["ctx_states"], data["actions"], data["branch"],
                   data.get("future_actions"))


def build_bifurcation_set(cfg: GameConfig, n: int = 1000, k: int = 4, seed: int = 0,
                          bots: tuple[str, ...] = ("random", "greedy"), n_envs: int = 256,
                          horizon: int = 16) -> BifurcationSet:
    """Tire des épisodes (flux aléatoires du split « test ») jusqu'à avoir n bifurcations dont
    le contexte de k images est réel (t >= k-1) et dont les `horizon` actions suivantes existent
    (t + horizon <= T), puis en garde n, tirées sans remise."""
    T = cfg.max_steps
    if horizon < 1 or k + horizon > T:
        raise ValueError("horizon incompatible avec la longueur d'épisode")
    per_bot = -(-n // len(bots))
    per_episode = max(1, (T - k + 1 - horizon) // (2 * cfg.spoke_len))   # bifurcations par épisode (≈)
    n_episodes = 2 * -(-per_bot // per_episode) + 2             # marge x2 : la collecte est bon marché
    ctx, act, br, fut = [], [], [], []
    for i, bot in enumerate(bots):
        env_seed, bot_seed = stream_seeds(seed, "test", i)
        d = collect_episodes(cfg, bot, n_episodes, env_seed, bot_seed, n_envs)
        e_idx, t_idx = np.nonzero(d["branch"] >= 0)
        keep = (t_idx >= k - 1) & (t_idx + horizon <= T)
        e_idx, t_idx = e_idx[keep], t_idx[keep]
        steps = t_idx[:, None] + np.arange(-k + 1, 1)[None, :]
        ctx.append(d["states"][e_idx[:, None], steps])
        act.append(d["actions"][e_idx, t_idx])
        br.append(d["branch"][e_idx, t_idx])
        fut.append(d["actions"][e_idx[:, None], t_idx[:, None] + np.arange(horizon)[None, :]])
    ctx, act, br, fut = np.concatenate(ctx), np.concatenate(act), np.concatenate(br), np.concatenate(fut)
    if len(act) < n:
        raise RuntimeError(f"seulement {len(act)} bifurcations trouvées pour {n} demandées")
    pick = np.sort(np.random.default_rng(stream_seeds(seed, "test", 99)[0]).choice(len(act), n, replace=False))
    return BifurcationSet(cfg, ctx[pick], act[pick], br[pick], fut[pick])


def count_duplicates(a: np.ndarray, b: np.ndarray) -> int:
    """Nombre de lignes de `a` (n, ...) qui figurent aussi dans `b` : détecte les fuites
    d'états entre entraînement et test. Les lignes sont comparées exactement."""
    def keys(x):
        x = np.ascontiguousarray(x.reshape(len(x), -1))
        return {row.tobytes() for row in x}

    return len(keys(a) & keys(b))
