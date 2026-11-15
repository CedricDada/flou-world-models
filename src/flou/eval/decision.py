"""Extension « décision », niveau 1 : le modèle estime-t-il bien un risque ? (hypothèses H0e, H4a, H4b, H4c)

Question : le flou change-t-il la réponse à « l'agent va-t-il toucher l'ennemi ? ». On fait prédire au modèle les k
images qui suivent un état de bifurcation (rollout autorégressif existant, actions futures du bot), on LIT le contact
dans les images prédites, et on compare au coût exact.

Articles qui guident ces choix (lus par extraction ciblée, voir docs/EXTENSION_DECISION.md) :

* PETS, Chua et al. 2018, arXiv:1805.12114, §5.2 et §7.2 : propager l'ESPÉRANCE de la prédiction (ici : la régression,
  dont la sortie idéale est l'image moyenne) est la base de comparaison de l'échantillonnage de trajectoires (ici : la
  diffusion et les jetons, dont on moyenne m tirages) ; de petits biais s'y accumulent ; le choix du modèle compte plus
  que la méthode de propagation.
* Objective Mismatch, Lambert et al. 2020, arXiv:2002.04523, §4.1 : précision du modèle et qualité de la décision sont
  deux objectifs distincts ; H4 peut donc être réfutée.

NON issu d'un article (notre raisonnement, que H4a/H4b testent) : un lecteur d'image LINÉAIRE appliqué à l'image
moyenne p·A + (1-p)·B rend exactement le bon risque à un pas ; le flou ne devrait nuire qu'à un lecteur à SEUIL, ou
quand des images floues sont réinjectées en entrée (k > 1).

Définitions
    coût de contact à l'horizon k   nombre de pas, parmi les k suivants, où l'agent chevauche l'ennemi. À k = 1 c'est
                                    la probabilité de contact. On prend un NOMBRE de pas et non « au moins un contact »
                                    parce que c'est ce que la récompense du jeu compte (-1 par pas de contact) et parce
                                    que cette quantité est linéaire : C* = p·C_A + (1-p)·C_B.
    exactitude                      le prochain état de bifurcation n'arrive que 2·spoke_len pas plus tard (6 par
                                    défaut) : pour k <= 5 un seul tirage de branche compte, C* est exact.
    état à risque ambigu            C_A != C_B : le coût dépend de la branche. C'est sur eux qu'on mesure l'erreur.
    lecteur doux                    part de « couleur de l'ennemi » sous la boîte de l'agent : affine en l'image.
    lecteur à seuil                 le même, seuillé à 0,5 (contact : oui ou non).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st

from ..data.dataset import BifurcationSet
from ..game.config import GameConfig
from ..game.dynamics import advance, boxes_overlap, is_branch_state
from ..game.render import render
from ..game.state import State
from .classifier import as_float
from .predictors import Batch, Predictor
from .probe import color_mask
from .rollout import autoregressive_rollout

READERS = ("soft", "hard")


def max_exact_horizon(cfg: GameConfig) -> int:
    """Plus grand horizon pour lequel un seul tirage de branche compte après un état de bifurcation."""
    return 2 * cfg.spoke_len - 1


# ---------------------------------------------------------------------------------------
# Lecture du contact dans une image
# ---------------------------------------------------------------------------------------
class ContactReader:
    """Lit « l'agent touche-t-il l'ennemi ? » dans des images (N, H, W, 3), uint8 ou flottants dans [0, 1].

    1. Agent : centre de gravité des pixels de sa couleur (comme la sonde analytique) -> coin de sa boîte.
    2. Ennemi : il ne vit que sur son rail (2·spoke_len + 1 positions). Pour chaque position j, e_j = moyenne, sur les
       pixels de la boîte NON couverts par l'agent (l'agent est dessiné par-dessus), de
           s(x) = (f(x) - f(fond)) / (f(ennemi) - f(fond)),   f(c) = R - (G + B) / 2,
       qui vaut 1 pour la couleur de l'ennemi et ≈ 0 pour le fond et l'obstacle. s est AFFINE en l'image.
    3. Contact doux = somme des e_j sur les positions dont la boîte chevauche celle de l'agent, bornée à [0, 1].
       Contact à seuil = (contact doux >= 0,5).

    Sur une image moyenne p·A + (1-p)·B, e_j vaut la probabilité que l'ennemi soit en j : le contact doux est alors
    exactement la probabilité de contact (c'est l'argument de linéarité que H4b teste). Hypothèse de palette : sur le
    rail, seul l'ennemi est « rouge » au sens de f ; le contrôle H0e (`check_reader`) le vérifie sur de vraies images.
    """

    def __init__(self, cfg: GameConfig, tol: float = 0.12, min_pixels: int = 1):
        self.cfg, self.tol, self.min_pixels = cfg, tol, min_pixels
        f = lambda c: float(c[0]) - (float(c[1]) + float(c[2])) / 2.0
        self._f_bg = f(cfg.color_bg) / 255.0
        self._f_scale = (f(cfg.color_enemy) - f(cfg.color_bg)) / 255.0
        if abs(self._f_scale) < 0.2:
            raise ValueError("palette incompatible avec le lecteur de contact (ennemi trop peu « rouge »)")
        hx, hy = cfg.hub
        self.rail_x = [hx + j * cfg.enemy_step for j in range(-cfg.spoke_len, cfg.spoke_len + 1)]
        self.rail_y = hy

    # -- positions ---------------------------------------------------------------------------
    def locate(self, imgs: np.ndarray, color, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coin haut-gauche (x, y) entier de l'objet de couleur `color` et de côté `size`, et masque « trouvé »."""
        imgs = np.asarray(imgs)
        n = len(imgs)
        if imgs.dtype == np.uint8:
            # chemin rapide (le planificateur lit des millions d'images) : on filtre d'abord sur le canal le plus
            # éloigné du fond, puis on ne vérifie les deux autres canaux que sur les pixels retenus. Même règle que
            # `color_mask` : chaque canal à moins de tol de la couleur.
            c = np.round(np.asarray(color, dtype=np.float64))
            lim = self.tol * 255.0 + 1e-9
            lo = np.clip(np.ceil(c - lim), 0, 255).astype(np.uint8)
            hi = np.clip(np.floor(c + lim), 0, 255).astype(np.uint8)
            ch = int(np.argmax(np.abs(c - np.asarray(self.cfg.color_bg, dtype=np.float64))))
            plane = imgs[..., ch]
            ni, yi, xi = np.nonzero((plane >= lo[ch]) & (plane <= hi[ch]))
            px = imgs[ni, yi, xi]
            ok = ((px >= lo) & (px <= hi)).all(axis=1)
            ni, yi, xi = ni[ok], yi[ok], xi[ok]
            cnt = np.bincount(ni, minlength=n)
            sx = np.bincount(ni, weights=xi, minlength=n)
            sy = np.bincount(ni, weights=yi, minlength=n)
        else:
            m = color_mask(imgs, np.asarray(color, dtype=np.float64) / 255.0, self.tol)
            cnt = m.sum(axis=(1, 2))
            sx = m.sum(axis=1) @ np.arange(m.shape[2])
            sy = m.sum(axis=2) @ np.arange(m.shape[1])
        found = cnt >= self.min_pixels
        safe = np.where(found, cnt, 1)
        cx, cy = sx / safe, sy / safe
        h = (size - 1) / 2.0
        x = np.where(found, np.round(cx - h), 0).astype(np.int32)
        y = np.where(found, np.round(cy - h), 0).astype(np.int32)
        return x, y, found

    def agent(self, imgs: np.ndarray):
        return self.locate(imgs, self.cfg.color_agent, self.cfg.agent_size)

    def gem(self, imgs: np.ndarray):
        return self.locate(imgs, self.cfg.color_gem, self.cfg.gem_size)

    # -- contact -----------------------------------------------------------------------------
    def soft(self, imgs: np.ndarray, agent=None) -> tuple[np.ndarray, np.ndarray]:
        """(contact doux (N,) dans [0, 1], agent trouvé (N,)). `agent` = (x, y, trouvé) si déjà lu."""
        imgs = np.asarray(imgs)
        ax, ay, found = self.agent(imgs) if agent is None else agent
        cfg = self.cfg
        A, E, y0 = cfg.agent_size, cfg.enemy_size, self.rail_y
        x_lo, x_hi = self.rail_x[0], self.rail_x[-1] + E
        band = imgs[:, y0:y0 + E, x_lo:x_hi]
        band = band.astype(np.float32) / 255.0 if band.dtype == np.uint8 else as_float(band).astype(np.float32)
        s = (band[..., 0] - (band[..., 1] + band[..., 2]) / 2.0 - self._f_bg) / self._f_scale      # (N, E, largeur)
        ys = np.arange(y0, y0 + E)[None, :, None]
        xs = np.arange(x_lo, x_hi)[None, None, :]
        under_agent = ((xs >= ax[:, None, None]) & (xs < ax[:, None, None] + A)
                       & (ys >= ay[:, None, None]) & (ys < ay[:, None, None] + A))
        vis = ~under_agent
        out = np.zeros(len(imgs), dtype=np.float64)
        for xj in self.rail_x:
            sl = slice(xj - x_lo, xj - x_lo + E)
            v = vis[:, :, sl]
            e = (s[:, :, sl] * v).sum(axis=(1, 2)) / np.maximum(v.sum(axis=(1, 2)), 1)
            out += np.where(boxes_overlap(ax, ay, A, xj, y0, E), e, 0.0)
        return np.where(found, np.clip(out, 0.0, 1.0), 0.0), found

    def hard(self, imgs: np.ndarray, agent=None) -> tuple[np.ndarray, np.ndarray]:
        c, found = self.soft(imgs, agent)
        return (c >= 0.5).astype(np.float64), found

    def read(self, imgs: np.ndarray, reader: str = "soft", agent=None):
        if reader not in READERS:
            raise ValueError(f"lecteur inconnu : {reader} (choix : {READERS})")
        return self.soft(imgs, agent) if reader == "soft" else self.hard(imgs, agent)


# ---------------------------------------------------------------------------------------
# Vérité : coût de contact exact par branche
# ---------------------------------------------------------------------------------------
def branch_paths(bset: BifurcationSet, k: int) -> dict:
    """Les deux futurs possibles (branche A, branche B) des k pas qui suivent chaque état de bifurcation, avec les
    actions futures du bot. Renvoie {"states_a", "states_b" : (N, k, 10) int16 ; "hit_a", "hit_b" : (N, k) bool}."""
    cfg = bset.cfg
    if k < 1 or k > max_exact_horizon(cfg):
        raise ValueError(f"horizon {k} hors de [1, {max_exact_horizon(cfg)}] : au-delà, une seconde bifurcation "
                         "intervient et le coût n'est plus une fonction d'un seul tirage")
    if bset.horizon < k:
        raise ValueError(f"le jeu de test ne porte que {bset.horizon} actions futures")
    s0 = bset.last_state()
    if not is_branch_state(s0).all():
        raise ValueError("tous les états doivent être des états de bifurcation")
    out = {}
    for name, b in (("a", 1), ("b", 0)):
        cur, states, hits = s0, [], []
        for i in range(k):
            br = np.where(is_branch_state(cur), b, -1)
            cur, ev = advance(cur, bset.future_actions[:, i], br, cfg)
            states.append(cur.to_array())
            hits.append(ev.hit_enemy)
        out[f"states_{name}"] = np.stack(states, axis=1)
        out[f"hit_{name}"] = np.stack(hits, axis=1)
    return out


def true_costs(bset: BifurcationSet, k: int, paths: dict | None = None) -> dict:
    """Coût de contact exact, cumulé sur 1..k pas. Renvoie des tableaux (N, k) :
    cost_a, cost_b (par branche), cost (espérance p·A + (1-p)·B), ambiguous (cost_a != cost_b)."""
    paths = paths or branch_paths(bset, k)
    ca = paths["hit_a"].cumsum(axis=1).astype(np.float64)
    cb = paths["hit_b"].cumsum(axis=1).astype(np.float64)
    p = bset.cfg.p
    return {"cost_a": ca, "cost_b": cb, "cost": p * ca + (1 - p) * cb, "ambiguous": ca != cb}


def select_states(bset: BifurcationSet, ks, n_other: int | None = None, seed: int = 0) -> np.ndarray:
    """Indices des états utiles au niveau 1 : tous ceux à risque ambigu pour au moins un k de `ks`, plus `n_other`
    états non ambigus tirés au hasard (par défaut : autant), qui servent à mesurer les fausses alertes."""
    amb = true_costs(bset, max(ks))["ambiguous"][:, [k - 1 for k in ks]].any(axis=1)
    idx_amb = np.nonzero(amb)[0]
    rest = np.nonzero(~amb)[0]
    n_other = len(idx_amb) if n_other is None else n_other
    rng = np.random.default_rng(seed)
    pick = rng.choice(rest, min(n_other, len(rest)), replace=False) if len(rest) and n_other > 0 else rest[:0]
    return np.sort(np.concatenate([idx_amb, pick]))


# ---------------------------------------------------------------------------------------
# Contrôle H0e : le lecteur retrouve-t-il l'événement de contact du jeu sur de VRAIES images ?
# ---------------------------------------------------------------------------------------
def check_reader(bset: BifurcationSet, k: int | None = None, threshold: float = 0.99) -> dict:
    """H0e. Sur les vraies images des deux branches (pas 1..k), compare la lecture du contact à `hit_enemy`."""
    cfg = bset.cfg
    k = max_exact_horizon(cfg) if k is None else k
    paths = branch_paths(bset, k)
    reader = ContactReader(cfg)
    agree, soft_err, n, found, n_hit = 0, 0.0, 0, 0, 0
    for name in ("a", "b"):
        for i in range(k):
            imgs = render(State.from_array(paths[f"states_{name}"][:, i]), cfg)
            truth = paths[f"hit_{name}"][:, i]
            c, f = reader.soft(imgs)
            agree += int(((c >= 0.5) == truth).sum())
            soft_err = max(soft_err, float(np.abs(c - truth).max()))
            found += int(f.sum())
            n += len(truth)
            n_hit += int(truth.sum())
    rate = agree / n
    return {"n_images": n, "n_contacts": n_hit, "agreement": rate, "soft_max_abs_err": soft_err,
            "agent_found": found / n, "threshold": threshold, "horizon": k, "pass": bool(rate >= threshold)}


# ---------------------------------------------------------------------------------------
# Mesure d'un prédicteur
# ---------------------------------------------------------------------------------------
def predicted_costs(pred: Predictor, bset: BifurcationSet, k: int, m: int = 1, seed: int = 0,
                    batch_size: int = 250) -> dict:
    """Coût de contact lu dans m rollouts prédits de k pas. Renvoie des tableaux (N, k) cumulés, moyennés sur les m
    tirages : "soft", "hard" ; et "agent_found" (k,) : part des images où l'agent a été retrouvé."""
    reader = ContactReader(bset.cfg)
    n = len(bset)
    soft, hard, found = np.zeros((n, k)), np.zeros((n, k)), np.zeros(k)
    for j in range(m):
        frames = autoregressive_rollout(pred, bset, k, seed=seed + 7919 * j, batch_size=batch_size)
        for i in range(k):
            c, f = reader.soft(frames[:, i])
            soft[:, i] += c
            hard[:, i] += c >= 0.5
            found[i] += f.mean()
    return {"soft": soft.cumsum(axis=1) / m, "hard": hard.cumsum(axis=1) / m, "agent_found": found / m}


def _err(x: np.ndarray) -> dict:
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "se": float("nan")}
    return {"n": int(n), "mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")}


def evaluate_risk(pred: Predictor, bset: BifurcationSet, ks=(1, 3, 5), m: int = 8, seed: int = 0,
                  batch_size: int = 250) -> dict:
    """Erreur d'estimation du coût de contact, par horizon k et par lecteur.

    m : tirages par état (générateurs : 8 par défaut, d'après PETS §5.1 qui en utilise 20 ; régression : 1, sa sortie
    est déterministe). E = moyenne de |Ĉ - C*| ; `amb` : états à risque ambigu (la mesure des hypothèses H4) ;
    `other` : états non ambigus (fausses alertes ou contacts manqués, où la branche ne change rien)."""
    kmax = max(ks)
    truth = true_costs(bset, kmax)
    got = predicted_costs(pred, bset, kmax, m, seed, batch_size)
    out = {"model": getattr(pred, "name", str(pred)), "p": bset.cfg.p, "n_states": len(bset), "m": m,
           "ks": [int(k) for k in ks], "per_k": {}}
    for k in ks:
        amb = truth["ambiguous"][:, k - 1]
        star = truth["cost"][:, k - 1]
        row = {"n_ambiguous": int(amb.sum()), "agent_found": float(got["agent_found"][:k].mean())}
        for r in READERS:
            e = np.abs(got[r][:, k - 1] - star)
            row[r] = {"amb": _err(e[amb]), "other": _err(e[~amb]),
                      "bias_amb": float((got[r][:, k - 1] - star)[amb].mean()) if amb.any() else float("nan")}
        out["per_k"][str(k)] = row
    return out


def risk_line(res: dict) -> str:
    parts = []
    for k in res["ks"]:
        r = res["per_k"][str(k)]
        parts.append(f"k={k} doux {r['soft']['amb']['mean']:.3f} seuil {r['hard']['amb']['mean']:.3f} "
                     f"(n={r['n_ambiguous']})")
    return f"{res['model']:<34} m={res['m']}  " + " | ".join(parts)


# ---------------------------------------------------------------------------------------
# Valeurs attendues « si le modèle était parfait » (k = 1, états ambigus)
# ---------------------------------------------------------------------------------------
def expected_error_sampler(p: float, q: float, m: int) -> float:
    """E|X/m - p| avec X ~ Binomiale(m, q) : erreur moyenne d'un échantillonneur NET qui tire la branche A avec la
    probabilité q (q = p : calibré), quand on moyenne m tirages. Par symétrie, la valeur est la même que le contact
    ait lieu dans la branche A ou dans la branche B."""
    x = np.arange(m + 1)
    return float((st.binom.pmf(x, m, q) * np.abs(x / m - p)).sum())


def expected_error_mean(p: float, reader: str) -> float:
    """Erreur d'un régresseur idéal (image moyenne), moyennée sur les états ambigus supposés également répartis entre
    « contact dans A seulement » et « contact dans B seulement » : 0 pour le lecteur doux ; pour le lecteur à seuil,
    la lecture vaut 0 ou 1 selon que le poids de la branche à contact dépasse 0,5."""
    if reader == "soft":
        return 0.0
    e_a = abs((p >= 0.5) - p)               # contact dans A seulement : C* = p
    e_b = abs(((1 - p) >= 0.5) - (1 - p))   # contact dans B seulement : C* = 1 - p
    return float((e_a + e_b) / 2)


# ---------------------------------------------------------------------------------------
# Prédicteurs factices de rollout (bancs d'essai du niveau 1 : la bonne réponse est connue)
# ---------------------------------------------------------------------------------------
class BranchRollout:
    """Rejoue les deux futurs exacts d'un jeu de test sur `horizon` pas (<= 5).

    q = None   « régression idéale » : rend p·A + (1-p)·B à chaque pas (image moyenne parfaitement propagée) ;
    q = float  « échantillonneur net » : tire la branche A avec la probabilité q au pas 0 et s'y tient ensuite
               (q = p : générateur parfait ; q != p : biais de calibration, par exemple celui d'Euler)."""

    def __init__(self, bset: BifurcationSet, horizon: int, q: float | None = None, name: str | None = None):
        self.cfg, self.q = bset.cfg, q
        paths = branch_paths(bset, horizon)
        self.sa, self.sb = paths["states_a"], paths["states_b"]
        self._pick = np.zeros(len(bset), dtype=bool)
        self.name = name or ("regression-ideale-propagee" if q is None else f"echantillonneur-net(q={q:.3f})")

    def predict(self, batch: Batch, rng: np.random.Generator) -> np.ndarray:
        idx, i = batch.idx, batch.step
        a = render(State.from_array(self.sa[idx, i]), self.cfg)
        b = render(State.from_array(self.sb[idx, i]), self.cfg)
        if self.q is None:
            p = self.cfg.p
            return p * (a.astype(np.float32) / 255.0) + (1 - p) * (b.astype(np.float32) / 255.0)
        if i == 0:
            self._pick[idx] = rng.random(len(idx)) < self.q
        return np.where(self._pick[idx][:, None, None, None], a, b)
