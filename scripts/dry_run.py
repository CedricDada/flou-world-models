#!/usr/bin/env python
"""RÉPÉTITION GÉNÉRALE : simule les 25 runs avec des prédicteurs factices et fait tourner toute l'analyse.

Aucun entraînement, aucun résultat scientifique : des effets sont plantés à la main (voir flou/eval/dryrun.py)
pour vérifier que l'évaluateur, les verdicts, les tables et les figures reconnaissent ce qu'on y a mis.
Tout est écrit sous « SIMULATION » (runs_dryrun/, reports/dryrun/), jamais dans runs/ ni reports/.

L'extension « décision » (H4, H5 ; docs/EXTENSION_DECISION.md) est répétée aussi, sur les 9 runs du cœur, avec un
planificateur réduit (peu de parties, peu de suites) : le contrôle H0f complet se lance avec validate_decision.py.

Exemples
    uv run scripts/dry_run.py                       # 25 runs, 1 000 états de test par p (quelques minutes)
    uv run scripts/dry_run.py --no-decision         # sans l'extension « décision » (plus rapide)
    uv run scripts/dry_run.py --states 300          # plus rapide
    uv run scripts/dry_run.py --out /tmp/essai
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

from flou.eval import analysis as A
from flou.eval import decision as D
from flou.eval import decision_analysis as DA
from flou.eval import decision_instrument as DI
from flou.eval import planning as P
from flou.eval import dryrun, instrument
from flou.eval import report as RP
from flou.train import runs as R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", type=int, default=1000, help="états de bifurcation par valeur de p")
    ap.add_argument("--steps", type=int, default=20_000, help="pas nominaux du plan (n'influe que sur les noms/budgets)")
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("."), help="racine : runs_dryrun/, data_dryrun/, reports/dryrun/")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-decision", action="store_true", help="ne pas répéter l'extension « décision » (H4, H5)")
    ap.add_argument("--decision-states", type=int, default=1500, help="états du jeu de test du niveau 1")
    ap.add_argument("--plan-games", type=int, default=8, help="parties du planificateur réduit")
    ap.add_argument("--plan-steps", type=int, default=50, help="pas joués par partie (planificateur réduit)")
    args = ap.parse_args()

    data, runs_dir, rep = args.out / "data_dryrun", args.out / "runs_dryrun", args.out / "reports" / "dryrun"
    plan = R.build_plan(steps=args.steps)
    t0 = time.time()
    # ρmax doit venir de l'instrument : on fabrique d'abord les jeux de test, on règle ρmax, puis on simule les runs
    from flou.data.dataset import build_bifurcation_set
    from flou.game import GameConfig
    for p in sorted({r.p for r in plan}):
        cfg = GameConfig().with_p(p)
        bs = build_bifurcation_set(cfg, n=args.states, seed=args.seed)
        bs.save(R.data_dir_for(data, p) / "test_bifurcation.npz", {"config": cfg.to_dict()})
    files = sorted(data.glob("p*/test_bifurcation.npz"))
    inst = instrument.validate_instrument(files, noise_sigma=args.noise, seed=args.seed)
    instrument.save_report(inst, rep / "instrument.json")
    print(f"instrument : ρmax = {inst['rho_max']:.4f}  H0a {'réussi' if inst['h0a']['pass'] else 'ÉCHOUÉ'}")
    if inst["avertissement_taille"]:
        print(inst["avertissement_taille"])

    rho = inst["rho_max"]
    for i, r in enumerate(plan, 1):
        bset = dryrun.BifurcationSet.load(R.test_file(data, r.p))
        dryrun.simulate_run(r, bset, rho, runs_dir)
        print(f"  [{i:>2}/{len(plan)}] {r.run_id}", flush=True)

    # ---- extension « décision » : contrôles H0e/H0f puis niveaux 1 et 2 simulés sur les runs du cœur --------------
    dinst, drecs, precs = None, [], []
    if not args.no_decision:
        p_core = 0.3
        cfg = GameConfig().with_p(p_core)
        dset = build_bifurcation_set(cfg, n=args.decision_states, seed=args.seed + 1000)
        dset.save(R.data_dir_for(data, p_core) / "test_decision.npz", {"config": cfg.to_dict()})
        pc = P.PlanConfig(candidates=32, samples=2)
        print(f"extension « décision » : contrôles H0e et H0f (planificateur réduit : {args.plan_games} parties de "
              f"{args.plan_steps} pas, J = {pc.candidates}, m = {pc.samples})", flush=True)
        dinst = DI.validate_decision(dset, seed=args.seed, pc=pc, n_games=args.plan_games, game_steps=args.plan_steps)
        DI.save_report(dinst, rep / "decision_instrument.json")
        print(f"  H0e {'réussi' if dinst['h0e']['pass'] else 'ÉCHOUÉ'}, H0f {'réussi' if dinst['h0f']['pass'] else 'ÉCHOUÉ'}")
        sub = dset.subset(D.select_states(dset, (1, 3, 5), seed=args.seed))
        core = [r for r in plan if r.group == "core"]
        for r in core:
            dryrun.simulate_decision(r, sub, runs_dir)
            if r.family in ("regression", "diffusion"):
                dryrun.simulate_planning(r, cfg, runs_dir, pc, args.plan_games, args.seed, args.plan_steps)
            print(f"  décision : {r.run_id}", flush=True)
        drecs, precs = DA.load_decision(runs_dir), DA.load_planning(runs_dir)

    recs = A.load_records(runs_dir)
    ver = A.verdicts(recs, inst)
    ver.update(DA.verdicts(drecs, precs, dinst, recs))
    ver["_simulation"] = True
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "verdicts.json").write_text(json.dumps(ver, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    ver.pop("_simulation")
    table = A.model_error_table(recs)
    A.write_csv(table, rep / "table_erreurs.csv")
    if drecs:
        A.write_csv(DA.risk_table(drecs), rep / "table_risque.csv")
        A.write_csv(DA.planning_table(precs, dinst), rep / "table_planification.csv")
    figs = []
    if not args.no_figures:
        try:
            from flou.eval import analysis_figs as F
            figs = F.all_figures(recs, rep)
        except ImportError as e:
            print(f"figures ignorées ({e})")
    (rep / "report.md").write_text(
        RP.render_report(recs, ver, plan, runs_dir, True, 0.3, figs,
                         decision={"risk": drecs, "planning": precs, "instrument": dinst}), encoding="utf-8")

    print("\n*** SIMULATION : aucun résultat scientifique ***")
    for k, v in ver.items():
        print(f"{k:<20} {v['status']}")
    print(f"\n{len(plan)} runs simulés en {time.time() - t0:.0f} s -> {runs_dir}/ et {rep}/ (report.md)")


if __name__ == "__main__":
    main()
