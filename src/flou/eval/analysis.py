"""Analyse des résultats : enregistrements à plat, verdicts H0 à H3, tables, figures.

Entrée : le dossier `runs/` (spec.json, eval.json, compute.json par run). Sortie : un dictionnaire de
verdicts (JSON), des tables CSV/Markdown et des figures. NumPy, SciPy et (pour les figures) matplotlib
seulement : l'analyse se teste sans PyTorch avec des résultats synthétiques.

Statuts : pour une hypothèse, « confirmée », « réfutée », « non concluante », « non évaluable » (runs
manquants ou trop peu de graines) ; pour un contrôle, « réussi », « échoué », « non évaluable ». Un résultat
sous le seuil à 3 graines est « non concluant », jamais « pas d'effet » (chap. 7) : on montre les points bruts.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import stats as S

FAMILY_ORDER = ("regression", "diffusion", "tokens", "oracle")
FAMILY_LABEL = {"regression": "A. Régression", "diffusion": "B. Diffusion (EDM)", "tokens": "C. Jetons",
                "oracle": "Régresseur oracle"}


# ---------------------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------------------
def _get(d: dict, *keys, default=np.nan):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def flatten_run(ev: dict, compute: dict | None = None) -> list[dict]:
    """Un enregistrement par variante d'échantillonnage d'un run."""
    spec = ev["spec"]
    recs = []
    for v in ev["variants"]:
        s = v["summary"]
        ro = v.get("rollout") or {}
        name = v["variant"]
        n_steps = int(name.split("=")[1]) if name.startswith("n=") else None
        recs.append({
            "run_id": ev["run_id"], "group": spec["group"], "family": spec["family"], "p": spec["p"],
            "seed": spec["seed"], "K": spec.get("K"), "grid": spec.get("grid", 8), "variant": name, "n_steps": n_steps,
            "n": s["n"], "blur": s["blur"]["value"], "blur_lo": s["blur"]["ci_lo"], "blur_hi": s["blur"]["ci_hi"],
            "sharp": s["sharp"]["value"], "other": s["other"]["value"],
            "p_hat": s["p_hat"]["value"], "p_hat_n": s["p_hat"]["n"], "p_hat_lo": s["p_hat"]["ci_lo"],
            "p_hat_hi": s["p_hat"]["ci_hi"], "p_abs_err": s["p_abs_err"], "alpha_mean": s["alpha_mean"],
            "recall": s["recall"]["value"], "recall_lo": s["recall"]["ci_lo"], "recall_hi": s["recall"]["ci_hi"],
            "hallucination": s["hallucination"]["value"],
            "ratio_expected": s["mse"]["ratio_expected"], "ratio_realized": s["mse"]["ratio_realized"],
            "mse_per_pixel": s["mse"]["per_pixel_realized"],
            "off_palette": s["off_palette"], "block_excess": s["block_edge_excess"] - s["block_edge_excess_truth"],
            "track_h1": _first(ro.get("track_ok")), "track_last": _last(ro.get("track_ok")),
            "gem_last": _last(ro.get("gem_recall")), "mse_static_last": _last(ro.get("mse_static")),
            "track_curve": ro.get("track_ok"), "recall_curve": ro.get("gem_recall"),
            "leak": bool(v.get("chk_h0b", {}).get("leak", False)),
            "vq_recall": _get(ev, "vq", "gem_recall"), "vq_mse": _get(ev, "vq", "recon_mse"),
            "perplexity": _get(ev, "vq", "perplexity_ema"),
            "gpu_hours": _get(compute or {}, "gpu_hours"), "params": _get(compute or {}, "parameters"),
        })
    return recs


def _first(x):
    return x[0] if x else np.nan


def _last(x):
    return x[-1] if x else np.nan


def load_records(runs_dir: str | Path) -> list[dict]:
    recs = []
    for d in sorted(Path(runs_dir).iterdir()):
        f = d / "eval.json"
        if not f.exists():
            continue
        comp = d / "compute.json"
        recs += flatten_run(json.loads(f.read_text(encoding="utf-8")),
                            json.loads(comp.read_text(encoding="utf-8")) if comp.exists() else None)
    return recs


def select(recs: list[dict], **kw) -> list[dict]:
    def ok(r):
        for k, v in kw.items():
            if isinstance(v, float):
                if not (r.get(k) is not None and abs(r[k] - v) < 1e-9):
                    return False
            elif r.get(k) != v:
                return False
        return True
    return [r for r in recs if ok(r)]


def by_seed(recs: list[dict], key: str) -> dict[int, float]:
    return {r["seed"]: r[key] for r in recs}


def _verdict(status: str, **evidence) -> dict:
    return {"status": status, **evidence}


def _na(why: str) -> dict:
    return _verdict("non évaluable", raison=why)


def _spearman(x, y) -> float:
    """Corrélation de rang ; nan si l'une des séries est constante (indéfinie, sans avertissement)."""
    from scipy.stats import spearmanr
    if len(set(np.round(x, 12))) < 2 or len(set(np.round(y, 12))) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _pts(recs: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in sorted(recs, key=lambda r: r["seed"])]


# ---------------------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------------------
def verdicts(recs: list[dict], instrument: dict | None = None, p_core: float = 0.3, min_n_h2b: int = 1000) -> dict:
    out: dict[str, dict] = {}

    # H0a : instrument validé sans entraînement
    if instrument is None:
        out["H0a"] = _na("reports/instrument.json absent : lancer scripts/validate_instrument.py")
    else:
        h = instrument["h0a"]
        out["H0a"] = _verdict("réussi" if h["pass"] else "échoué", alpha_mae=h["alpha_mae_mix"],
                              pures_bien_classees=h["pure_ok"], rho_max=instrument["rho_max"],
                              bruit=instrument["noise_sigma"])

    reg = select(recs, group="core", family="regression", p=p_core)
    dif3 = select(recs, group="core", family="diffusion", p=p_core, n_steps=3)
    dif1 = select(recs, group="core", family="diffusion", p=p_core, n_steps=1)
    dif20 = select(recs, group="core", family="diffusion", p=p_core, n_steps=20)

    # H0b : régression au plancher de Bayes
    if not reg:
        out["H0b"] = _na("pas de run de régression du cœur")
    else:
        rat = _pts(reg, "ratio_expected")
        ok = all(1.00 <= r <= 1.10 for r in rat)
        leak = any(r["leak"] for r in reg)
        out["H0b"] = _verdict("réussi" if ok and not leak else "échoué", rapports_en_esperance=rat,
                              rapports_tires=_pts(reg, "ratio_realized"), fuite_detectee=leak,
                              lecture="sous-apprentissage si > 1,10 ; fuite d'information si le rapport tiré est nettement < 1")

    # H0c, H0d et contrôle de diffusion à p = 0
    def blur_ctrl(recs_):
        if not recs_:
            return None
        b = recs_[0]["blur"]
        return _verdict("réussi" if b < 0.05 else "échoué", flou=b, ic95=[recs_[0]["blur_lo"], recs_[0]["blur_hi"]])

    for hid, sel in (("H0c", select(recs, group="control", family="oracle")),
                     ("H0d", select(recs, group="control", family="regression", p=0.0)),
                     ("H0d-diffusion(n=1)", select(recs, group="control", family="diffusion", p=0.0, n_steps=1))):
        out[hid] = blur_ctrl(sel) or _na("run de contrôle manquant")

    # H1 : régression floue, diffusion n = 3 nette (contrôle positif)
    if len(reg) < 1 or not dif3:
        out["H1"] = _na("runs du cœur manquants")
    else:
        rb, db = np.mean(_pts(reg, "blur")), np.mean(_pts(dif3, "blur"))
        ds = np.mean(_pts(dif3, "sharp"))
        refuted = rb < 0.30 or db > 0.10
        confirmed = rb > 0.50 and ds > 0.90
        status = "réfutée" if refuted else "confirmée" if confirmed else "non concluante"
        out["H1"] = _verdict(status, flou_regression=_pts(reg, "blur"), flou_diffusion_n3=_pts(dif3, "blur"),
                             nettes_diffusion_n3=_pts(dif3, "sharp"), moyenne_flou_regression=float(rb),
                             moyenne_nettes_diffusion_n3=float(ds), n_graines=len(reg),
                             regle="confirmée si flou_reg > 50 % et nettes_diff(n=3) > 90 % ; réfutée si flou_reg < 30 % ou flou_diff(n=3) > 10 %")

    # H2a : n = 1 plus floue que n = 3 d'au moins 10 points (comparaison appariée sur les mêmes débruiteurs)
    seeds = sorted(set(by_seed(dif1, "blur")) & set(by_seed(dif3, "blur")))
    if len(seeds) < 2:
        out["H2a"] = _na("moins de 2 graines de diffusion évaluées à n = 1 et n = 3")
    else:
        b1, b3 = by_seed(dif1, "blur"), by_seed(dif3, "blur")
        a, b = [b1[s] for s in seeds], [b3[s] for s in seeds]
        cmp_ = S.compare_groups(a, b, paired=True)
        if cmp_.diff >= 0.10 and cmp_.significant:
            status = "confirmée"
        elif cmp_.diff < 0.10:
            status = "réfutée"
        else:
            status = "non concluante"
        out["H2a"] = _verdict(status, ecart_moyen_points=100 * cmp_.diff, seuil_apparie_en_sd=cmp_.threshold,
                              ratio_en_sd=cmp_.ratio, flou_n1=a, flou_n3=b, n_graines=len(seeds),
                              regle="confirmée si écart >= 10 points et > 2,48 s_d ; réfutée si écart < 10 points")

    # H2b : calibration à n = 3
    if not dif3:
        out["H2b"] = _na("pas de diffusion n = 3")
    else:
        err = _pts(dif3, "p_abs_err")
        nsharp = [r["p_hat_n"] for r in sorted(dif3, key=lambda r: r["seed"])]
        enough = all(n >= min_n_h2b for n in nsharp)
        if not enough:
            status = "non concluante"
        elif all(e < 0.05 for e in err):
            status = "confirmée"
        elif all(e >= 0.05 for e in err):
            status = "réfutée"
        else:
            status = "non concluante"
        curve = {}
        for n, rr in (("n=1", dif1), ("n=3", dif3), ("n=20", dif20)):
            if rr:
                curve[n] = _pts(rr, "p_hat")
        out["H2b"] = _verdict(status, erreur_absolue=err, images_nettes=nsharp, p_hat_par_n=curve, p=p_core,
                              regle=f"confirmée si |p̂-p| < 0,05 pour chaque graine avec >= {min_n_h2b} images nettes ; "
                                    "réfutée si >= 0,05 pour chaque graine")

    # H3a : rappel du petit objet K = 64 contre K = 1024
    k64 = select(recs, family="tokens", K=64, p=p_core, grid=8)
    k256 = select(recs, family="tokens", K=256, p=p_core, grid=8, group="core")
    k1024 = select(recs, family="tokens", K=1024, p=p_core, grid=8)
    if len(k64) < 2 or len(k1024) < 2:
        out["H3a"] = _na("moins de 2 graines à K = 64 ou à K = 1024")
    else:
        a, b = _pts(k64, "recall"), _pts(k1024, "recall")
        cmp_ = S.compare_groups(a, b)
        if cmp_.diff < 0 and cmp_.significant:
            status = "confirmée"
        elif cmp_.diff >= 0:
            status = "réfutée"
        else:
            status = "non concluante"
        means = {64: float(np.mean(a)), 256: float(np.mean(_pts(k256, "recall"))) if k256 else None, 1024: float(np.mean(b))}
        mono = None if means[256] is None else bool(means[64] <= means[256] <= means[1024])
        out["H3a"] = _verdict(status, rappel_K64=a, rappel_K1024=b, rappel_K256=_pts(k256, "recall") if k256 else None,
                              ecart_moyen=cmp_.diff, ratio_en_sd_poolees=cmp_.ratio, seuil=cmp_.threshold,
                              ordre_monotone_des_moyennes=mono,
                              regle="confirmée si rappel(64) < rappel(1024) de plus de 2,27 écarts-types poolés")

    # H3b (exploratoire) : rappel contre erreur quadratique en pixels comme prédicteur de la qualité à l'horizon 16
    tok = [r for r in recs if r["family"] == "tokens" and abs(r["p"] - p_core) < 1e-9 and r["grid"] == 8
           and r["group"] in ("core", "k_axis") and not np.isnan(r["track_last"])]
    if len(tok) < 4:
        out["H3b"] = _na("moins de 4 points (K x graine) avec rollout")
    else:
        q = [r["track_last"] for r in tok]
        r_rec = _spearman([r["recall"] for r in tok], q)
        r_mse = _spearman([r["mse_per_pixel"] for r in tok], q)
        out["H3b"] = _verdict("exploratoire (sans seuil)", n_points=len(tok),
                              spearman_rappel_qualite16=r_rec, spearman_mse_qualite16=r_mse,
                              points=[{"K": r["K"], "seed": r["seed"], "rappel": r["recall"], "mse": r["mse_per_pixel"],
                                       "qualite16": r["track_last"]} for r in tok])
    return out


# ---------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------
def _mean(recs: list[dict], key: str) -> float:
    v = [r[key] for r in recs if r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return float(np.mean(v)) if v else float("nan")


def model_error_table(recs: list[dict], p: float = 0.3) -> list[dict]:
    """Table modèle x type d'erreur (taxonomie du dossier, §1.9), moyennes sur les graines du cœur / de l'axe K.

    a flou · b effondrement de mode (p̂ à moins de 0,02 de 0 ou 1 alors que 0 < p < 1) · autre (image hors segment)
    c perte du petit objet (1 - rappel) · d accumulation d'erreur (chute du suivi entre pas 1 et pas 16)
    e artefacts (excès de discontinuité aux frontières de blocs ; part de pixels hors palette)."""
    rows = []
    groups: dict[str, list[dict]] = {}
    for r in recs:
        if abs(r["p"] - p) > 1e-9 or r["group"] not in ("core", "k_axis", "control", "grid"):
            continue
        if r["group"] == "control" and r["family"] != "oracle":
            continue
        label = FAMILY_LABEL[r["family"]]
        if r["family"] == "diffusion":
            label += f" n={r['n_steps']}"
        if r["family"] == "tokens":
            label += f" K={r['K']}" + (f" grille {r['grid']}" if r["grid"] != 8 else "")
        groups.setdefault(label, []).append(r)
    for label, rr in groups.items():
        collapse = [(not np.isnan(r["p_hat"])) and (r["p_hat"] < 0.02 or r["p_hat"] > 0.98) and 0 < r["p"] < 1 for r in rr]
        rows.append({"modèle": label, "graines": len(rr), "flou": _mean(rr, "blur"), "autre": _mean(rr, "other"),
                     "effondrement_de_mode": float(np.mean(collapse)), "perte_petit_objet": 1 - _mean(rr, "recall"),
                     "chute_du_suivi_1_16": _mean(rr, "track_h1") - _mean(rr, "track_last"),
                     "excès_blocs": _mean(rr, "block_excess"), "hors_palette": _mean(rr, "off_palette"),
                     "mse_sur_plancher": _mean(rr, "ratio_expected"), "heures_gpu": _mean(rr, "gpu_hours")})
    order = {"A": 0, "B": 1, "C": 2, "R": 3}
    rows.sort(key=lambda r: (order.get(r["modèle"][0], 9), r["modèle"]))
    return rows


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def markdown_table(rows: list[dict], digits: int = 3) -> str:
    if not rows:
        return "(vide)\n"
    cols = list(rows[0])
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def compute_report(recs: list[dict]) -> list[dict]:
    """Heures de GPU réellement consommées, par run (une ligne par run : les variantes d'échantillonnage
    partagent le même entraînement)."""
    seen, rows = set(), []
    for r in recs:
        if r["run_id"] in seen:
            continue
        seen.add(r["run_id"])
        rows.append({"run": r["run_id"], "groupe": r["group"], "paramètres": r["params"], "heures_gpu": r["gpu_hours"]})
    return rows


def calibration_by_n(recs: list[dict], p: float = 0.3) -> list[dict]:
    rows = []
    for n in sorted({r["n_steps"] for r in recs if r["family"] == "diffusion" and r["n_steps"]}):
        rr = select(recs, family="diffusion", group="core", p=p, n_steps=n)
        if rr:
            rows.append({"n": n, "graines": len(rr), "p_hat_moyen": _mean(rr, "p_hat"),
                         "erreur_absolue_moyenne": _mean(rr, "p_abs_err"), "flou": _mean(rr, "blur"),
                         "nettes": _mean(rr, "sharp")})
    return rows
