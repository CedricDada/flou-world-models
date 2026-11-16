"""Rapport Markdown des résultats : verdicts, tables, runs manquants, limites. NumPy seulement.

Un rapport issu de runs SIMULÉS (`flou.eval.dryrun`) porte une bannière explicite et ne doit jamais être lu
comme un résultat : il sert à vérifier que l'analyse tourne et reconnaît les effets qu'on y a plantés.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import analysis as A
from . import decision_analysis as DA

HYPOTHESES = {
    "H0a": "Instrument : sans entraînement, α̂ exact et images pures bien classées",
    "H0b": "Régression au plancher de Bayes (rapport MSE/plancher dans [1,00 ; 1,10], sans fuite)",
    "H0c": "Régresseur oracle (reçoit la branche) : moins de 5 % de flou",
    "H0d": "Régression à p = 0 : moins de 5 % de flou",
    "H0d-diffusion(n=1)": "Diffusion à p = 0, n = 1 : moins de 5 % de flou",
    "H1": "Régression floue, diffusion (n = 3) nette",
    "H2a": "n = 1 plus floue que n = 3 d'au moins 10 points (comparaison appariée)",
    "H2b": "Calibration à n = 3 : |p̂ − p| < 0,05 à chaque graine",
    "H3a": "Le rappel du petit objet croît avec K (K = 64 contre K = 1024)",
    "H3b": "Rappel ou MSE-pixel comme prédicteur de la qualité à l'horizon 16 (exploratoire)",
    **DA.HYPOTHESES,
}

LIMITS = (
    "3 graines par cellule du cœur : un écart sous le seuil est « non concluant », jamais « pas d'effet » ; les points "
    "bruts sont donnés dans verdicts.json.",
    "Les seuils sont des règles de décision à très peu de graines, pas des preuves. Le cœur compte k = 3 comparaisons "
    "simultanées fixées avant les runs (H1, H2a, H3a ; docs/DECISIONS.md, C4) : 3,23 s non apparié (H3a), 4,42 s_d apparié (H2a) ; "
    "l'extension (H4a, H5a) reste à 2,27 s.",
    "ρmax est réglé sur un bruit synthétique : la part « autre » dépend de ce choix (sensibilité × 0,5 / × 1 / × 2 dans "
    "instrument.json).",
    "Les axes p et la grille 16 × 16 n'ont qu'une graine : exploratoire.",
    "Aucune relecture humaine n'est incluse : la validité du classifieur s'estime avec review.csv (scripts/human_review.py).",
)


def missing_runs(plan, runs_dir: str | Path) -> list[str]:
    """Identifiants du plan sans eval.json."""
    root = Path(runs_dir)
    return [r.run_id for r in plan if not (root / r.run_id / "eval.json").exists()]


def is_simulated(runs_dir: str | Path) -> bool:
    """Vrai si au moins un eval.json, decision.json ou planning.json porte la marque SIMULATION (flou.eval.dryrun)."""
    for name in ("eval.json", "decision.json", "planning.json"):
        for f in Path(runs_dir).glob(f"*/{name}"):
            try:
                if json.loads(f.read_text(encoding="utf-8")).get("SIMULATION"):
                    return True
            except json.JSONDecodeError:
                continue
    return False


def _cell(text: str) -> str:
    """Un « | » dans une cellule Markdown casserait le tableau (|p̂ − p| dans H2b)."""
    return str(text).replace("|", "\\|")


def _short(v: dict) -> str:
    for k in ("regle", "lecture", "raison"):
        if k in v:
            return str(v[k])
    return ""


def render_report(recs: list[dict], verdict_dict: dict, plan=None, runs_dir: str | Path | None = None,
                  simulated: bool = False, p: float = 0.3, figures: list[Path] | None = None,
                  decision: dict | None = None) -> str:
    """`decision` (facultatif) : {"risk": enregistrements du niveau 1, "planning": ceux du niveau 2,
    "instrument": contenu de decision_instrument.json} -> section « Extension décision »."""
    L: list[str] = []
    if simulated:
        L += ["> **SIMULATION — AUCUN RÉSULTAT SCIENTIFIQUE.** Ces chiffres viennent de prédicteurs factices (flou.eval.dryrun) "
              "dans lesquels des effets ont été plantés à la main ; ils vérifient la chaîne d'analyse, pas les modèles.", ""]
    L += ["# Rapport d'analyse", ""]
    n_runs = len({r["run_id"] for r in recs})
    L.append(f"{n_runs} runs évalués, {len(recs)} variantes d'échantillonnage.")
    if plan is not None and runs_dir is not None:
        miss = missing_runs(plan, runs_dir)
        if miss:
            L.append(f"\n**Runs du plan sans résultat ({len(miss)}/{len(plan)}) :** " + ", ".join(miss) + ". "
                     "Les hypothèses qui en dépendent sont « non évaluables » ou « non concluantes ».")
    L += ["", "## Verdicts", "", "| Hypothèse | Énoncé | Statut | Règle / raison |", "|---|---|---|---|"]
    for k, v in verdict_dict.items():
        L.append(f"| {k} | {_cell(HYPOTHESES.get(k, ''))} | **{v['status']}** | {_cell(_short(v))} |")
    L += ["", "## Types d'erreur par modèle (taxonomie du dossier, §1.9)", "",
          A.markdown_table(A.model_error_table(recs, p))]
    cal = A.calibration_by_n(recs, p)
    if cal:
        L += ["## Calibration de la diffusion selon le nombre de pas n", "", A.markdown_table(cal)]
    comp = A.compute_report(recs)
    if comp:
        total = sum(r["heures_gpu"] for r in comp if r["heures_gpu"] == r["heures_gpu"])
        L += ["## Calcul consommé", "", f"Total : {total:.1f} h de GPU sur {len(comp)} runs.", "", A.markdown_table(comp)]
    if decision is not None:
        L += DA.report_section(decision.get("risk", []), decision.get("planning", []), decision.get("instrument"), p)
    if figures:
        L += ["## Figures", ""] + [f"- {f.name}" for f in figures] + [""]
    L += ["## Limites", ""] + [f"- {x}" for x in LIMITS + (DA.LIMITS if decision is not None else ())] + [""]
    return "\n".join(L)
