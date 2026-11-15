#!/usr/bin/env python
"""Analyse des résultats : verdicts H0 à H5, tables CSV/Markdown, figures, rapport.

Exemples
    uv run scripts/analyze.py                         # runs/ -> reports/
    uv run scripts/analyze.py --runs runs --p-core 0.3
    uv run scripts/analyze.py --runs runs_dryrun      # runs SIMULÉS -> reports/dryrun/ (bannière SIMULATION)

Sorties dans --out : verdicts.json, report.md, table_erreurs.csv/.md, calibration_par_n.csv, calcul.csv, fig_*.png ;
si l'extension « décision » a été mesurée (scripts/evaluate_decision.py) : table_risque.csv, table_planification.csv.
Les hypothèses H4 et H5 sont « non évaluables » tant que ces mesures manquent : le reste de l'analyse n'en dépend pas.
Des runs simulés ne sont jamais écrits dans reports/ (il est réservé aux vrais résultats).
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from flou.eval import analysis as A
from flou.eval import decision_analysis as DA
from flou.eval import report as RP
from flou.train import runs as R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--instrument", type=Path, default=Path("reports/instrument.json"))
    ap.add_argument("--decision-instrument", type=Path, default=Path("reports/decision_instrument.json"))
    ap.add_argument("--out", type=Path, default=None, help="défaut : reports (ou reports/dryrun si runs simulés)")
    ap.add_argument("--p-core", type=float, default=0.3)
    ap.add_argument("--min-sharp", type=int, default=1000, help="images nettes minimales pour conclure sur H2b")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if not args.runs.is_dir():
        raise SystemExit(f"{args.runs} introuvable : lancer scripts/train.py / scripts/run_plan.py "
                         "(ou scripts/dry_run.py pour une répétition simulée)")
    recs = A.load_records(args.runs)
    if not recs:
        raise SystemExit(f"aucun eval.json dans {args.runs}/*/ : lancer scripts/evaluate.py")
    simulated = RP.is_simulated(args.runs)
    out = args.out or Path("reports/dryrun" if simulated else "reports")
    if simulated and out.resolve() == Path("reports").resolve():
        raise SystemExit("runs SIMULÉS : refus d'écrire dans reports/ (réservé aux vrais résultats) ; utiliser reports/dryrun")
    out.mkdir(parents=True, exist_ok=True)

    inst = json.loads(args.instrument.read_text(encoding="utf-8")) if args.instrument.exists() else None
    if simulated and inst is None:
        alt = out / "instrument.json"
        inst = json.loads(alt.read_text(encoding="utf-8")) if alt.exists() else None
    ver = A.verdicts(recs, inst, p_core=args.p_core, min_n_h2b=args.min_sharp)

    # extension « décision » (H0e, H0f, H4, H5) : facultative, « non évaluable » si rien n'a été mesuré
    dinst_path = out / "decision_instrument.json" if simulated else args.decision_instrument
    dinst = json.loads(dinst_path.read_text(encoding="utf-8")) if dinst_path.exists() else None
    drecs, precs = DA.load_decision(args.runs), DA.load_planning(args.runs)
    ver.update(DA.verdicts(drecs, precs, dinst, recs, args.p_core))
    if drecs:
        A.write_csv(DA.risk_table(drecs, args.p_core), out / "table_risque.csv")
    if precs:
        A.write_csv(DA.planning_table(precs, dinst, args.p_core), out / "table_planification.csv")
    ver["_simulation"] = simulated
    (out / "verdicts.json").write_text(json.dumps(ver, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    ver.pop("_simulation")

    table = A.model_error_table(recs, args.p_core)
    A.write_csv(table, out / "table_erreurs.csv")
    (out / "table_erreurs.md").write_text(A.markdown_table(table), encoding="utf-8")
    cal = A.calibration_by_n(recs, args.p_core)
    if cal:
        A.write_csv(cal, out / "calibration_par_n.csv")
    A.write_csv(A.compute_report(recs), out / "calcul.csv")

    figs = []
    if not args.no_figures:
        try:
            from flou.eval import analysis_figs as F
            figs = F.all_figures(recs, out, args.p_core)
        except ImportError as e:
            print(f"figures ignorées ({e})")
    (out / "report.md").write_text(
        RP.render_report(recs, ver, R.build_plan(), args.runs, simulated, args.p_core, figs,
                         decision={"risk": drecs, "planning": precs, "instrument": dinst}), encoding="utf-8")

    if simulated:
        print("*** SIMULATION : aucun résultat scientifique ***")
    for k, v in ver.items():
        print(f"{k:<20} {v['status']}")
    print(f"\n{len(recs)} enregistrements -> {out}/ (report.md, verdicts.json, tables, {len(figs)} figures)")


if __name__ == "__main__":
    main()
