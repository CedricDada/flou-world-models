"""Extension « décision », niveau 2 : un agent qui planifie DANS le modèle (hypothèses H0f, H5a, H5b).

À chaque pas du vrai jeu, le planificateur tire J suites de H actions, fait IMAGINER au modèle les images qui
suivraient, lit une récompense dans ces images, joue la première action de la meilleure suite, puis recommence.
Le modèle sert donc de simulateur : c'est l'usage qui justifie le mot « world model ».

Articles qui guident ces choix (lus par extraction ciblée, voir docs/EXTENSION_DECISION.md) :

* PETS, Chua et al. 2018, arXiv:1805.12114 : contrôle prédictif (MPC), on n'exécute que la PREMIÈRE action puis on
  replanifie (§6, Alg. 1) ; particules = tirages du modèle (§5.1, 20 particules) ; propager l'espérance avec une seule
  particule est leur base de comparaison (§5.2) : c'est le rôle que joue ici la régression (m = 1, image moyenne),
  face aux générateurs (m tirages).
* PlaNet, Hafner et al. 2019, arXiv:1811.04551 : replanification à chaque pas (§2) ; réglages CEM H = 12, 10 itérations,
  1 000 suites, 100 élites (annexe A) ; une seule trajectoire par suite candidate suffit chez eux (§2) ; ils planifient
  en espace latent SANS générer d'images. Nous générons des images : d'où un tirage simple de J = 64 suites (une seule
  itération de CEM, sans élites) et H = 8. Leur Table 1 compare à un planificateur qui dispose du vrai simulateur :
  c'est notre contrôle H0f (`SimulatorModel`).
* Objective Mismatch, Lambert et al. 2020, arXiv:2002.04523, §4.1 : un modèle plus précis ne donne pas forcément un
  meilleur contrôle ; H5a peut être réfutée et H5b est exploratoire.

NON issu d'un article (nos choix, à discuter) :
* la récompense est LUE DANS LES PIXELS imaginés (contact : `ContactReader` ; ramassage : la boîte de l'agent imaginé
  atteint le petit objet réel) : elle est donc supposée connue, alors que PlaNet l'apprend ;
* un terme de rapprochement -λ·distance(agent imaginé, petit objet)/taille en fin d'horizon : sans lui, le petit objet
  est trop loin pour être atteint en H pas et le planificateur ne fait qu'éviter l'ennemi (mesuré sur le vrai jeu) ;
* lecteur de contact « doux » par défaut : c'est le plus favorable à la régression (voir H4b), donc le test le plus
  sévère pour H5a.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from ..game.config import N_ACTIONS, GameConfig
from ..game.dynamics import advance, boxes_overlap, is_branch_state
from ..game.env import VecMiniGame
from ..game.render import render
from ..game.state import State
from .decision import READERS, ContactReader
from .predictors import Batch, Predictor, to_uint8


@dataclass(frozen=True)
class PlanConfig:
    """Réglages du planificateur. Valeurs par défaut = docs/EXTENSION_DECISION.md §5 ([choix])."""
    candidates: int = 64        # J suites d'actions tirées uniformément à chaque décision
    horizon: int = 8            # H pas imaginés
    samples: int = 4            # m tirages du modèle par suite (générateurs) ; 1 pour un modèle déterministe
    lam: float = 0.5            # poids du terme de rapprochement du petit objet
    reader: str = "soft"        # lecteur de contact : "soft" ou "hard"
    k_ctx: int = 4              # images de contexte données au modèle
    max_rows: int = 2048        # lignes (partie x suite x tirage) traitées à la fois : borne la mémoire

    def __post_init__(self):
        if self.reader not in READERS:
            raise ValueError(f"lecteur inconnu : {self.reader}")
        if min(self.candidates, self.horizon, self.samples, self.k_ctx) < 1:
            raise ValueError("candidates, horizon, samples et k_ctx doivent être >= 1")

    def to_dict(self) -> dict:
        return asdict(self)

    def predictions_per_decision(self) -> int:
        """Images que le modèle doit produire pour UNE décision d'UNE partie."""
        return self.candidates * self.samples * self.horizon


class Planner:
    """Planificateur par tirage de suites d'actions (« random shooting »), sur l'interface `Predictor`.

    Si le prédicteur possède `begin_plan(states)` (c'est le cas des simulateurs factices, qui ont besoin de l'état
    exact), le planificateur l'appelle avant d'imaginer ; un modèle appris ne voit que les images."""

    def __init__(self, pred: Predictor, cfg: GameConfig, pc: PlanConfig = PlanConfig()):
        self.pred, self.cfg, self.pc = pred, cfg, pc
        self.reader = ContactReader(cfg)
        self.n_predictions = 0
        self.n_lost = 0             # images imaginées où l'agent n'a pas été retrouvé
        self.n_read = 0

    def act(self, ctx: np.ndarray, rng: np.random.Generator, states: State | None = None) -> np.ndarray:
        """ctx : (N, k, H, W, 3) uint8, images RÉELLES les plus récentes. Renvoie une action par partie (N,)."""
        n = len(ctx)
        per_game = self.pc.candidates * self.pc.samples
        chunk = max(1, self.pc.max_rows // per_game)
        out = np.zeros(n, dtype=np.int32)
        for lo in range(0, n, chunk):
            sl = slice(lo, min(n, lo + chunk))
            out[sl] = self._act_chunk(ctx[sl], rng, None if states is None else states[sl])
        return out

    def _act_chunk(self, ctx: np.ndarray, rng: np.random.Generator, states: State | None) -> np.ndarray:
        cfg, pc, rd = self.cfg, self.pc, self.reader
        g, J, m, H = len(ctx), pc.candidates, pc.samples, pc.horizon
        rows = g * J * m
        seqs = rng.integers(0, N_ACTIONS, size=(g, J, H))
        seq_rows = np.repeat(seqs, m, axis=1).reshape(rows, H)
        # situation réelle : agent et petit objet lus dans la dernière image RÉELLE
        ax0, ay0, _ = rd.agent(ctx[:, -1])
        gx0, gy0, gem_seen = rd.gem(ctx[:, -1])
        rep = lambda x: np.repeat(x, J * m, axis=0)
        ax, ay, gx, gy, gem_seen = rep(ax0), rep(ay0), rep(gx0), rep(gy0), rep(gem_seen)
        keep_ctx = getattr(self.pred, "needs_context", True)        # les simulateurs factices n'en ont pas besoin
        cur = rep(ctx) if keep_ctx else ctx[:0]
        if hasattr(self.pred, "begin_plan"):
            if states is None:
                raise ValueError("ce prédicteur est un simulateur : il lui faut les états exacts (states=)")
            self.pred.begin_plan(states[np.repeat(np.arange(g), J * m)])
        A, G = cfg.agent_size, cfg.gem_size
        ret = np.zeros(rows)
        picked = np.zeros(rows, dtype=bool)
        idx = np.arange(rows)
        for h in range(H):
            y = to_uint8(self.pred.predict(Batch(idx, cur, seq_rows[:, h], h), rng))
            x_, y_, found = rd.agent(y)
            ax, ay = np.where(found, x_, ax), np.where(found, y_, ay)     # agent perdu : dernière position connue
            c, _ = rd.read(y, pc.reader, (ax, ay, np.ones(rows, dtype=bool)))
            ret -= c
            pick = gem_seen & ~picked & boxes_overlap(ax, ay, A, gx, gy, G)
            ret += pick
            picked |= pick
            self.n_lost += int((~found).sum())
            if keep_ctx:
                cur = np.concatenate([cur[:, 1:], y[:, None]], axis=1)
        self.n_read += rows * H
        self.n_predictions += rows * H
        if pc.lam:
            d = (np.abs(2 * ax + A - 2 * gx - G) + np.abs(2 * ay + A - 2 * gy - G)) / 2.0 / cfg.size
            ret -= pc.lam * np.where(picked | ~gem_seen, 0.0, d)
        score = ret.reshape(g, J, m).mean(axis=2)
        best = score.argmax(axis=1)
        return seqs[np.arange(g), best, 0].astype(np.int32)


def play(pred: Predictor, cfg: GameConfig, pc: PlanConfig = PlanConfig(), n_games: int = 50, seed: int = 0,
         plan_seed: int | None = None, steps: int | None = None, progress=None) -> dict:
    """Fait jouer le planificateur dans le VRAI jeu. Les parties (états initiaux, tirages de branches) ne dépendent
    que de `seed` : tous les modèles évalués avec la même graine jouent exactement les mêmes parties (comparaison
    appariée). `plan_seed` : graine des suites d'actions et des tirages du modèle.

    Avant le pas k-1 d'une partie, le contexte répète la première image (comme à l'entraînement, voir
    `flou.data.dataset.TransitionData`)."""
    T = cfg.max_steps if steps is None else min(steps, cfg.max_steps)
    env = VecMiniGame(cfg, n_games, seed=seed)
    obs, _ = env.reset()
    ctx = np.repeat(obs[:, None], pc.k_ctx, axis=1)
    rng = np.random.default_rng(seed + 1 if plan_seed is None else plan_seed)
    planner = Planner(pred, cfg, pc)
    score, hits, picks = np.zeros(n_games), np.zeros(n_games), np.zeros(n_games)
    t0 = time.time()
    for t in range(T):
        a = planner.act(ctx, rng, env.state)
        obs, r, _, _, info = env.step(a)
        score += r
        hits += info["events"].hit_enemy
        picks += info["events"].pickup
        ctx = np.concatenate([ctx[:, 1:], obs[:, None]], axis=1)
        if progress is not None:
            progress(t + 1, T, time.time() - t0)
    sd = lambda x: float(x.std(ddof=1)) if n_games > 1 else float("nan")
    return {"model": getattr(pred, "name", str(pred)), "p": cfg.p, "n_games": n_games, "steps": T, "seed": seed,
            "config": pc.to_dict(),
            "score": float(score.mean()), "score_sd": sd(score),
            "contacts": float(hits.mean()), "contacts_sd": sd(hits),
            "pickups": float(picks.mean()), "pickups_sd": sd(picks),
            "agent_lost": planner.n_lost / max(planner.n_read, 1),
            "predictions": planner.n_predictions, "seconds": time.time() - t0,
            "per_game": {"score": score.tolist(), "contacts": hits.tolist(), "pickups": picks.tolist()}}


def play_bot(name: str, cfg: GameConfig, n_games: int = 50, seed: int = 0, steps: int | None = None) -> dict:
    """Score d'un bot sur LES MÊMES parties que `play` (même graine) : repère pour lire les scores."""
    from ..bots import make_bot
    T = cfg.max_steps if steps is None else min(steps, cfg.max_steps)
    env = VecMiniGame(cfg, n_games, seed=seed, render_obs=False)
    env.reset()
    bot = make_bot(name, cfg)
    bot.reset(n_games)
    rng = np.random.default_rng(seed + 1)
    score, hits, picks = np.zeros(n_games), np.zeros(n_games), np.zeros(n_games)
    for _ in range(T):
        _, r, _, _, info = env.step(bot.act(env.state, rng))
        score += r
        hits += info["events"].hit_enemy
        picks += info["events"].pickup
    return {"model": f"bot-{name}", "n_games": n_games, "steps": T, "score": float(score.mean()),
            "contacts": float(hits.mean()), "pickups": float(picks.mean())}


def play_line(res: dict) -> str:
    return (f"{res['model']:<34} score {res['score']:6.2f}  contacts {res['contacts']:5.2f}  "
            f"ramassages {res['pickups']:5.2f}" + (f"  ({res['seconds']:.0f} s)" if "seconds" in res else ""))


# ---------------------------------------------------------------------------------------
# Simulateurs factices : le VRAI jeu utilisé comme modèle (contrôle H0f, bancs d'essai)
# ---------------------------------------------------------------------------------------
class SimulatorModel:
    """Le vrai jeu présenté comme un `Predictor`. Sert de plafond de référence (PlaNet, Table 1) et de banc d'essai.

    mode = "sample"  tire la branche A avec la probabilité q (q = p par défaut : simulateur exact) ;
    mode = "mean"    rend l'image MOYENNE p·A + (1-p)·B, parfaitement propagée : une « régression idéale » qui ne
                     souffrirait pas de recevoir ses propres images floues en entrée. Deux hypothèses suffisent : tous
                     les futurs de l'ennemi repassent par le centre du rail au même pas, où ils se confondent.
    fade             (mode "mean" seulement) facteur INVENTÉ, dans [0, 1], qui efface un peu plus l'ennemi à chaque pas
                     imaginé (intensité x fade^pas) : simule un flou qui s'aggrave quand on réinjecte des images
                     floues. Ne sert qu'à la répétition générale ; ce n'est la mesure d'aucun modèle.
    """

    def __init__(self, cfg: GameConfig, mode: str = "sample", q: float | None = None, fade: float = 1.0,
                 name: str | None = None):
        if mode not in ("sample", "mean"):
            raise ValueError(mode)
        self.cfg, self.mode, self.fade = cfg, mode, fade
        self.q = cfg.p if q is None else q
        self.name = name or ("vrai-jeu" if mode == "sample" and q is None else
                             f"vrai-jeu(q={self.q:.3f})" if mode == "sample" else
                             "vrai-jeu-moyenne" + (f"(fondu {fade:g})" if fade != 1.0 else ""))
        self._a: State | None = None
        self._b: State | None = None

    needs_context = False       # ce « modèle » lit l'état exact, pas les images : inutile de lui tenir un contexte

    def begin_plan(self, states: State) -> None:
        self._a, self._b = states.copy(), states.copy()

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        if self._a is None:
            raise RuntimeError("begin_plan(states) doit être appelé avant predict")
        cfg, act = self.cfg, np.asarray(batch.actions, dtype=np.int32)
        hub = is_branch_state(self._a)
        if self.mode == "sample":
            br = np.where(hub, (rng.random(len(act)) < self.q).astype(np.int32), -1)
            self._a, _ = advance(self._a, act, br, cfg)
            return render(self._a, cfg)
        self._a, _ = advance(self._a, act, np.where(hub, 1, -1), cfg)
        self._b, _ = advance(self._b, act, np.where(is_branch_state(self._b), 0, -1), cfg)
        # A et B ne diffèrent que par l'ennemi, donc que dans la bande du rail : on ne mélange que cette bande
        # (le résultat est quantifié sur 8 bits, comme le fait de toute façon le planificateur avant de relire).
        img_a = render(self._a, cfg)
        y0, y1 = cfg.hub[1], cfg.hub[1] + cfg.enemy_size
        band_a = img_a[:, y0:y1].astype(np.float32)
        band_b = render_band(self._b, cfg, y0, y1).astype(np.float32)
        mix = cfg.p * band_a + (1 - cfg.p) * band_b
        if self.fade != 1.0:
            w = self.fade ** (batch.step + 1)
            mix = w * mix + (1 - w) * render_band(self._a, cfg, y0, y1, enemy=False).astype(np.float32)
        img_a[:, y0:y1] = np.clip(np.round(mix), 0, 255).astype(np.uint8)
        return img_a


def render_band(s: State, cfg: GameConfig, y0: int, y1: int, enemy: bool = True) -> np.ndarray:
    """Lignes y0..y1 de l'image de l'état (avec ou sans l'ennemi). Passe par le rendu complet : simple et sûr."""
    if enemy:
        return render(s, cfg)[:, y0:y1]
    from ..game.dynamics import geometry
    from ..game.render import _draw_boxes
    img = np.broadcast_to(geometry(cfg).base, (s.n, cfg.size, cfg.size, 3)).copy()
    _draw_boxes(img, s.ox, s.oy, cfg.obstacle_size, cfg.color_obstacle)
    _draw_boxes(img, s.ax, s.ay, cfg.agent_size, cfg.color_agent)
    _draw_boxes(img, s.gx, s.gy, cfg.gem_size, cfg.color_gem)
    return img[:, y0:y1]


# ---------------------------------------------------------------------------------------
# Contrôle H0f
# ---------------------------------------------------------------------------------------
def check_planner(cfg: GameConfig, pc: PlanConfig = PlanConfig(), n_games: int = 50, seed: int = 0,
                  steps: int | None = None, max_contacts: float = 0.5, min_gain: float = 3.0, progress=None) -> dict:
    """H0f : avec le VRAI jeu comme modèle (et la lecture de la récompense dans les pixels), le planificateur doit
    éviter l'ennemi (<= `max_contacts` contacts par partie) et battre le bot aléatoire d'au moins `min_gain` points,
    sur les mêmes parties. Sinon le planificateur ou le lecteur est en cause, et rien ne peut être conclu sur un modèle."""
    res = play(SimulatorModel(cfg), cfg, pc, n_games, seed, steps=steps, progress=progress)
    rnd = play_bot("random", cfg, n_games, seed, steps)
    exp = play_bot("expert", cfg, n_games, seed, steps)
    scale = res["steps"] / cfg.max_steps        # les seuils valent pour une partie entière : proportionnels sinon
    gain = res["score"] - rnd["score"]
    ok = res["contacts"] <= max_contacts * scale and gain >= min_gain * scale
    return {"planner": {k: v for k, v in res.items() if k != "per_game"}, "random": rnd, "expert": exp,
            "gain_over_random": gain, "max_contacts": max_contacts * scale, "min_gain": min_gain * scale,
            "pass": bool(ok)}
