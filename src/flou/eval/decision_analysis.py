"""Analyse de l'extension « décision » : verdicts H0e, H0f, H4a, H4b, H4c, H5a, H5b et tables. NumPy et SciPy seulement.

Entrées (toutes facultatives : ce qui manque donne « non évaluable ») :
    reports/decision_instrument.json   contrôles H0e et H0f (scripts/validate_decision.py, sans GPU)
    runs/<run>/decision.json           niveau 1 : erreur d'estimation du coût de contact (scripts/evaluate_decision.py)
    runs/<run>/planning.json           niveau 2 : score du planificateur dans le vrai jeu (--level 2)

Règles écrites AVANT toute mesure d'un modèle (docs/EXTENSION_DECISION.md) ; les seuils sont des [choix] :
    H4a  lecteur à seuil, k = 1 : E(régression) - E(diffusion n=3) >= 0,10 ET > 2,27 écarts-types poolés (3 contre 3)
         -> confirmée ; écart < 0,10 -> réfutée ; sinon non concluante.   Guidé par PETS (arXiv:1805.12114) §5.2, §7.2.
    H4b  lecteur doux, k = 1 : E(régression) <= E(diffusion n=3) + 0,02 -> confirmée ; > + 0,05 -> réfutée ; entre les
         deux -> non concluante.   Notre raisonnement de linéarité (pas un article) ; PETS §7.2.
    H4c  exploratoire : courbes E_k, sans seuil.   PETS §5.2 (accumulation des biais).
    H5a  contacts par partie : régression - diffusion n=3 >= 0,5 ET > 2,27 écarts-types poolés -> confirmée ;
         écart < 0,5 -> réfutée.   PETS §7.2 ; prudence de Lambert et al. (arXiv:2002.04523) §4.1.
    H5b  exploratoire : corrélations de rang entre le score du planificateur et des mesures de qualité d'image.
         Lambert et al. §4.1, Fig. 3 : de -0,06 à 0,59 chez eux, avec plus de 1 000 modèles ; ici 6 à 9 points.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import analysis as A
from . import stats as S

H4A_GAP, H4B_OK, H4B_BAD, H5A_GAP = 0.10, 0.02, 0.05, 0.5

HYPOTHESES = {
    "H0e": "Lecteur de contact : sur de vraies images, il retrouve l'événement du jeu (accord >= 99 %)",
    "H0f": "Planificateur avec le vrai jeu comme modèle : <= 0,5 contact par partie et >= 3 points de plus que le bot aléatoire",
    "H4a": "Lecteur à seuil, 1 pas : la régression estime moins bien le risque que la diffusion (n = 3)",
    "H4b": "Lecteur doux, 1 pas : la régression estime le risque au moins aussi bien que la diffusion (n = 3)",
    "H4c": "Erreur d'estimation du coût de contact selon l'horizon k (exploratoire)",
    "H5a": "Planification : le modèle flou (régression) subit plus de contacts que le modèle net (diffusion n = 3)",
    "H5b": "Score du planificateur contre qualité d'image (exploratoire)",
}

LIMITS = (
    "Extension « décision » : la récompense est lue dans les pixels (supposée connue), le planificateur est un tirage "
    "simple de suites d'actions ; un autre planificateur ou une récompense apprise pourraient changer les conclusions.",
    "H5b porte sur 6 à 9 points : Lambert et al. (arXiv:2002.04523) trouvent des corrélations de -0,06 à 0,59 avec plus "
    "de 1 000 modèles ; une corrélation de rang sur si peu de points est indicative, rien de plus.",
)

MAIN_VARIANT = {"regression": "", "diffusion": "n=3", "tokens": "tau=1"}


# ---------------------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------------------
def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_decision(runs_dir: str | Path) -> list[dict]:
    """Un enregistrement par (run, variante) : erreurs moyennes sur les états à risque ambigu, par lecteur et par k."""
    recs = []
    for d in sorted(Path(runs_dir).iterdir()):
        js = _read(d / "decision.json")
        if not js:
            continue
        spec = js["spec"]
        for v in js["variants"]:
            r = v["risk"]
            rec = {"run_id": js["run_id"], "group": spec["group"], "family": spec["family"], "p": spec["p"],
                   "seed": spec["seed"], "K": spec.get("K"), "variant": v["variant"], "m": r["m"], "ks": r["ks"]}
            for k in r["ks"]:
                row = r["per_k"][str(k)]
                rec[f"soft_k{k}"] = row["soft"]["amb"]["mean"]
                rec[f"hard_k{k}"] = row["hard"]["amb"]["mean"]
                rec[f"other_soft_k{k}"] = row["soft"]["other"]["mean"]
                rec[f"n_amb_k{k}"] = row["n_ambiguous"]
                rec[f"found_k{k}"] = row["agent_found"]
            recs.append(rec)
    return recs


def load_planning(runs_dir: str | Path) -> list[dict]:
    recs = []
    for d in sorted(Path(runs_dir).iterdir()):
        js = _read(d / "planning.json")
        if not js:
            continue
        spec = js["spec"]
        for v in js["variants"]:
            pl = v["play"]
            recs.append({"run_id": js["run_id"], "group": spec["group"], "family": spec["family"], "p": spec["p"],
                         "seed": spec["seed"], "K": spec.get("K"), "variant": v["variant"],
                         "score": pl["score"], "contacts": pl["contacts"], "pickups": pl["pickups"],
                         "n_games": pl["n_games"], "reader": pl["config"]["reader"], "samples": pl["config"]["samples"],
                         "agent_lost": pl["agent_lost"], "seconds": pl.get("seconds", float("nan"))})
    return recs


def _core(recs: list[dict], family: str, p: float) -> list[dict]:
    return A.select(recs, group="core", family=family, p=p, variant=MAIN_VARIANT[family])


# ---------------------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------------------
def verdicts(drecs: list[dict], precs: list[dict], instrument: dict | None = None,
             eval_recs: list[dict] | None = None, p_core: float = 0.3) -> dict:
    out: dict[str, dict] = {}
    why_inst = "reports/decision_instrument.json absent : lancer scripts/validate_decision.py"

    # H0e, H0f : contrôles sans entraînement
    h0e = (instrument or {}).get("h0e")
    out["H0e"] = A._na(why_inst) if not h0e else A._verdict(
        "réussi" if h0e["pass"] else "échoué", accord=h0e["agreement"], images=h0e["n_images"],
        contacts=h0e["n_contacts"], erreur_max_lecteur_doux=h0e["soft_max_abs_err"])
    h0f = (instrument or {}).get("h0f")
    if not h0f:
        out["H0f"] = A._na(why_inst + " (sans --no-planner)")
    else:
        out["H0f"] = A._verdict("réussi" if h0f["pass"] else "échoué", contacts_par_partie=h0f["planner"]["contacts"],
                                score=h0f["planner"]["score"], score_bot_aleatoire=h0f["random"]["score"],
                                score_bot_expert=h0f["expert"]["score"], gain=h0f["gain_over_random"],
                                parties=h0f["planner"]["n_games"], reglages=h0f["planner"]["config"])

    reg, dif = _core(drecs, "regression", p_core), _core(drecs, "diffusion", p_core)

    # H4a : lecteur à seuil, k = 1
    key = "hard_k1"
    if len(reg) < 2 or len(dif) < 2 or key not in reg[0]:
        out["H4a"] = A._na("moins de 2 graines de régression ou de diffusion (n = 3) évaluées au niveau 1, k = 1")
    else:
        a, b = A._pts(reg, key), A._pts(dif, key)
        c = S.compare_groups(a, b)
        status = ("confirmée" if c.diff >= H4A_GAP and c.significant else
                  "réfutée" if c.diff < H4A_GAP else "non concluante")
        out["H4a"] = A._verdict(status, erreur_regression=a, erreur_diffusion_n3=b, ecart_moyen=c.diff,
                                ratio_en_sd_poolees=c.ratio, seuil=c.threshold,
                                regle=f"confirmée si écart >= {H4A_GAP:.2f} et > 2,27 écarts-types poolés ; "
                                      f"réfutée si écart < {H4A_GAP:.2f}")

    # H4b : lecteur doux, k = 1
    key = "soft_k1"
    if not reg or not dif or key not in reg[0]:
        out["H4b"] = A._na("pas de régression ou de diffusion (n = 3) évaluée au niveau 1, k = 1")
    else:
        a, b = A._pts(reg, key), A._pts(dif, key)
        gap = float(np.mean(a) - np.mean(b))
        status = "confirmée" if gap <= H4B_OK else "réfutée" if gap > H4B_BAD else "non concluante"
        out["H4b"] = A._verdict(status, erreur_regression=a, erreur_diffusion_n3=b, ecart_moyen=gap,
                                regle=f"confirmée si E(régression) <= E(diffusion) + {H4B_OK:.2f} ; réfutée si > "
                                      f"+ {H4B_BAD:.2f} ; entre les deux : non concluante")

    # H4c : exploratoire
    curves = {}
    for fam in ("regression", "diffusion", "tokens"):
        rr = _core(drecs, fam, p_core)
        if rr:
            curves[fam] = {f"k={k}": {"doux": float(np.mean(A._pts(rr, f"soft_k{k}"))),
                                      "seuil": float(np.mean(A._pts(rr, f"hard_k{k}")))} for k in rr[0]["ks"]}
    out["H4c"] = A._verdict("exploratoire (sans seuil)", erreur_moyenne_par_horizon=curves) if curves else \
        A._na("aucun run du cœur évalué au niveau 1")

    # H5a : contacts par partie du planificateur
    preg, pdif = _core(precs, "regression", p_core), _core(precs, "diffusion", p_core)
    if len(preg) < 2 or len(pdif) < 2:
        out["H5a"] = A._na("moins de 2 graines de régression ou de diffusion (n = 3) évaluées au niveau 2")
    else:
        a, b = A._pts(preg, "contacts"), A._pts(pdif, "contacts")
        c = S.compare_groups(a, b)
        status = ("confirmée" if c.diff >= H5A_GAP and c.significant else
                  "réfutée" if c.diff < H5A_GAP else "non concluante")
        out["H5a"] = A._verdict(status, contacts_regression=a, contacts_diffusion_n3=b, ecart_moyen=c.diff,
                                ratio_en_sd_poolees=c.ratio, seuil=c.threshold,
                                scores_regression=A._pts(preg, "score"), scores_diffusion_n3=A._pts(pdif, "score"),
                                lecteur=preg[0]["reader"],
                                regle=f"confirmée si écart >= {H5A_GAP:g} contact par partie et > 2,27 écarts-types "
                                      f"poolés ; réfutée si écart < {H5A_GAP:g}")

    # H5b : exploratoire, score du planificateur contre qualité d'image (variante principale de chaque run)
    pts = []
    by_run = {(r["run_id"], r["variant"]): r for r in (eval_recs or [])}
    for r in precs:
        if r["group"] != "core" or abs(r["p"] - p_core) > 1e-9 or r["variant"] != MAIN_VARIANT.get(r["family"]):
            continue
        e = by_run.get((r["run_id"], r["variant"]))
        if e is not None:
            pts.append({"run": r["run_id"], "score": r["score"], "contacts": r["contacts"], "nettes": e["sharp"],
                        "mse_pixel": e["mse_per_pixel"], "suivi_16": e["track_last"]})
    if len(pts) < 4:
        out["H5b"] = A._na("moins de 4 runs du cœur avec à la fois eval.json et planning.json")
    else:
        sc = [x["score"] for x in pts]
        out["H5b"] = A._verdict("exploratoire (sans seuil)", n_points=len(pts),
                                spearman_score_nettes=A._spearman([x["nettes"] for x in pts], sc),
                                spearman_score_mse_pixel=A._spearman([x["mse_pixel"] for x in pts], sc),
                                spearman_score_suivi_16=A._spearman([x["suivi_16"] for x in pts], sc), points=pts)
    return out


# ---------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------
def _label(r: dict) -> str:
    lab = A.FAMILY_LABEL[r["family"]]
    if r["family"] == "diffusion":
        lab += f" {r['variant']}"
    if r["family"] == "tokens":
        lab += f" K={r['K']}"
    return lab


def risk_table(drecs: list[dict], p: float = 0.3) -> list[dict]:
    """Erreur moyenne |Ĉ - C*| sur les états à risque ambigu, par modèle (moyenne sur les graines du cœur)."""
    groups: dict[str, list[dict]] = {}
    for r in drecs:
        if r["group"] == "core" and abs(r["p"] - p) < 1e-9:
            groups.setdefault(_label(r), []).append(r)
    rows = []
    for lab, rr in sorted(groups.items()):
        row = {"modèle": lab, "graines": len(rr), "tirages_m": rr[0]["m"]}
        for k in rr[0]["ks"]:
            row[f"doux_k{k}"] = float(np.mean([x[f"soft_k{k}"] for x in rr]))
            row[f"seuil_k{k}"] = float(np.mean([x[f"hard_k{k}"] for x in rr]))
        rows.append(row)
    return rows


def planning_table(precs: list[dict], instrument: dict | None = None, p: float = 0.3) -> list[dict]:
    rows = []
    h0f = (instrument or {}).get("h0f")
    if h0f:
        for name, key in (("bot aléatoire", "random"), ("bot expert", "expert"), ("planificateur, vrai jeu (H0f)", "planner")):
            x = h0f[key]
            rows.append({"joueur": name, "graines": 1, "parties": x["n_games"], "score": x["score"],
                         "contacts_par_partie": x["contacts"], "ramassages_par_partie": x["pickups"]})
    groups: dict[str, list[dict]] = {}
    for r in precs:
        if r["group"] == "core" and abs(r["p"] - p) < 1e-9:
            groups.setdefault("planificateur, " + _label(r), []).append(r)
    for lab, rr in sorted(groups.items()):
        rows.append({"joueur": lab, "graines": len(rr), "parties": rr[0]["n_games"],
                     "score": float(np.mean([x["score"] for x in rr])),
                     "contacts_par_partie": float(np.mean([x["contacts"] for x in rr])),
                     "ramassages_par_partie": float(np.mean([x["pickups"] for x in rr]))})
    return rows


def report_section(drecs: list[dict], precs: list[dict], instrument: dict | None, p: float = 0.3) -> list[str]:
    """Lignes Markdown ajoutées au rapport d'analyse."""
    L = ["## Extension « décision » (H4, H5)", "",
         "Hypothèses, seuils et articles qui les guident : `docs/EXTENSION_DECISION.md`.", ""]
    exp = (instrument or {}).get("expected")
    if exp:
        L += ["Erreur attendue si les modèles étaient parfaits (prédicteurs factices, vrai jeu) :", "",
              A.markdown_table(exp)]
    rt = risk_table(drecs, p)
    if rt:
        L += ["Niveau 1 — erreur d'estimation du coût de contact, états à risque ambigu :", "", A.markdown_table(rt)]
    pt = planning_table(precs, instrument, p)
    if pt:
        L += ["Niveau 2 — scores dans le vrai jeu (mêmes parties pour tous) :", "", A.markdown_table(pt, digits=2)]
    if not (rt or pt or exp):
        L += ["Aucune mesure de l'extension pour l'instant (scripts/validate_decision.py, scripts/evaluate_decision.py).", ""]
    return L
