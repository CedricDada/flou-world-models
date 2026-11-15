"""Rollouts de 16 pas : accumulation d'erreur (taxonomie (d)) et qualité à l'horizon 16 (H3b).

Que mesurer quand le futur est stochastique ? Seul l'ennemi dépend de la branche ; l'agent, l'obstacle
et le petit objet sont des fonctions des seules actions (vérifié par test : `tests/test_eval.py`).
La vérité terrain de ces trois éléments est donc UNIQUE, quelle que soit la suite de branches :
on les compare image par image, à chaque pas h, à l'image prédite autorégressivement.

Mesures par pas (tableaux de longueur H) :

* `track_ok`   part des rollouts où l'agent ET l'obstacle sont retrouvés à moins de `tol` pixels (3 px par
               défaut : les occultations décalent le centre de gravité jusqu'à ~2 px) par la sonde ;
* `gem_recall` le petit objet est présent à son emplacement exact ;
* `mse_static` erreur quadratique moyenne par pixel HORS du rail de l'ennemi (zone déterministe) ;
* `off_palette` part de pixels hors palette (dérive, bruit, flou cumulés).

« Qualité à l'horizon 16 » (H3b) = `track_ok` au dernier pas ; elle est exploratoire.
"""
from __future__ import annotations

import numpy as np

from ..data.dataset import BifurcationSet
from ..game.dynamics import advance, is_branch_state
from ..game.render import render
from .predictors import Batch, Predictor, to_uint8
from .probe import AnalyticProbe, position_errors, true_positions
from .recall import gem_recall, off_palette_fraction


def rollout_truth(bset: BifurcationSet, horizon: int | None = None, seed: int = 0):
    """Vérité terrain des `horizon` pas qui suivent chaque état de bifurcation.

    Au pas 0, la branche est celle du jeu de test ; aux bifurcations suivantes elle est tirée avec la
    probabilité p. Renvoie (frames (N, H, 64, 64, 3) uint8, états : liste de H objets State)."""
    h = bset.horizon if horizon is None else horizon
    if h < 1 or h > bset.horizon:
        raise ValueError(f"horizon {h} indisponible (le jeu de test porte {bset.horizon} actions futures)")
    cfg, rng = bset.cfg, np.random.default_rng(seed)
    s = bset.last_state()
    frames, states = [], []
    for i in range(h):
        br = np.where(is_branch_state(s), (rng.random(s.n) < cfg.p).astype(np.int32), -1)
        if i == 0:
            br = np.where(is_branch_state(s), bset.branch, -1)
        s, _ = advance(s, bset.future_actions[:, i], br, cfg)
        frames.append(render(s, cfg))
        states.append(s)
    return np.stack(frames, axis=1), states


def autoregressive_rollout(pred: Predictor, bset: BifurcationSet, horizon: int, seed: int = 0,
                           batch_size: int = 250) -> np.ndarray:
    """Images prédites (N, H, 64, 64, 3) uint8 : chaque image prédite est quantifiée sur 8 bits puis
    rebouclée en entrée, avec les actions futures du bot."""
    rng = np.random.default_rng(seed)
    ctx0 = bset.contexts()
    n = len(bset)
    out = np.zeros((n,) + (horizon,) + ctx0.shape[2:], dtype=np.uint8)
    for lo in range(0, n, batch_size):
        idx = np.arange(lo, min(n, lo + batch_size))
        ctx = ctx0[idx].copy()
        for i in range(horizon):
            y = to_uint8(pred.predict(Batch(idx, ctx, bset.future_actions[idx, i], i), rng))
            out[idx, i] = y
            ctx = np.concatenate([ctx[:, 1:], y[:, None]], axis=1)
    return out


def rollout_metrics(pred_frames: np.ndarray, truth_frames: np.ndarray, truth_states: list, bset: BifurcationSet,
                    tol: float = 3.0, probe=None) -> dict:
    """Mesures par pas d'un rollout prédit contre la vérité terrain. `probe` : sonde de position
    (par défaut la sonde analytique) ; elle doit renvoyer (N, 4, 2)."""
    cfg = bset.cfg
    probe = probe or AnalyticProbe(cfg)
    n, h = pred_frames.shape[:2]
    track, rec, mse, off, found = [], [], [], [], []
    for i in range(h):
        pf, tf, st = pred_frames[:, i], truth_frames[:, i], truth_states[i]
        err = position_errors(probe(pf), true_positions(st, cfg))      # (N, 4)
        agent, obst = err[:, 0], err[:, 1]
        ok = (agent <= tol) & (obst <= tol)                            # nan <= tol vaut False
        track.append(float(ok.mean()))
        found.append(float((~np.isnan(agent) & ~np.isnan(obst)).mean()))
        rec.append(float(gem_recall(pf, st.gx, st.gy, cfg).mean()))
        mask = np.ones(pf.shape[1:3], dtype=bool)
        mask[cfg.hub[1]:cfg.hub[1] + cfg.enemy_size, :] = False        # hors du rail (bande de l'ennemi)
        d = pf[:, mask].astype(np.int32) - tf[:, mask].astype(np.int32)
        mse.append(float((d ** 2).mean() / 255.0 ** 2))
        off.append(float(off_palette_fraction(pf, cfg).mean()))
    return {"horizon": h, "track_ok": track, "found": found, "gem_recall": rec, "mse_static": mse,
            "off_palette": off, "quality_last": track[-1]}


def evaluate_rollout(pred: Predictor, bset: BifurcationSet, horizon: int = 16, seed: int = 0, probe=None,
                     batch_size: int = 250) -> dict:
    truth, states = rollout_truth(bset, horizon, seed)
    pf = autoregressive_rollout(pred, bset, horizon, seed, batch_size)
    out = rollout_metrics(pf, truth, states, bset, probe=probe)
    out["model"] = getattr(pred, "name", str(pred))
    return out
