#!/usr/bin/env python
"""Exécute le plan de 25 runs dans l'ordre de priorité : entraînement puis évaluation, sans refaire ce qui est fini.

NON EXÉCUTÉ dans l'environnement de rédaction (pas de PyTorch) : lancer d'abord `uv run scripts/smoke_test.py --time`.

Ordre : cœur (9) et contrôles (3) d'abord, puis axe K (6), grille (1), axe p (6, sacrifié en premier si le calcul
manque). Chaque run s'exécute dans un sous-processus (une panne ou un manque de mémoire n'arrête pas le plan) ;
l'état est écrit dans runs/plan_status.json. Relancer la commande reprend où l'on s'est arrêté.

Exemples
    uv run scripts/run_plan.py                                  # tout, 20 000 pas par run
    uv run scripts/run_plan.py --steps 8000 --max-hours 10      # budget réduit, s'arrête proprement après 10 h
    uv run scripts/run_plan.py --groups core control            # seulement les 12 runs prioritaires
    uv run scripts/run_plan.py --skip-groups p_axis grid        # sacrifier d'avance ce qui est exploratoire
    uv run scripts/run_plan.py --dry                            # affiche ce qui serait lancé
Colab : la session peut s'interrompre ; relancer simplement la même commande (les checkpoints sont repris).
"""
import _bootstrap

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from flou.train import runs as R

HERE = Path(__file__).resolve().parent


def status_of(run_dir: Path) -> str:
    if (run_dir / "eval.json").exists():
        return "évalué"
    if (run_dir / "final.pt").exists():
        return "entraîné"
    if (run_dir / "ckpt.pt").exists():
        return "en cours"
    return "à faire"


def call(script: str, *args) -> int:
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    return subprocess.run(cmd).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--groups", nargs="+", choices=R.GROUPS, default=None, help="seulement ces groupes")
    ap.add_argument("--skip-groups", nargs="+", choices=R.GROUPS, default=[])
    ap.add_argument("--max-hours", type=float, default=None, help="durée maximale de CETTE exécution")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--instrument", type=Path, default=Path("reports/instrument.json"))
    ap.add_argument("--no-eval", action="store_true", help="n'évalue pas (par exemple ρmax pas encore validé)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    plan = [r for r in R.build_plan(steps=args.steps, batch_size=args.batch_size, lr=args.lr)
            if (args.groups is None or r.group in args.groups) and r.group not in args.skip_groups]
    if not args.no_eval and not args.instrument.exists() and not args.dry:
        raise SystemExit(f"{args.instrument} absent : lancer uv run scripts/validate_instrument.py avant l'évaluation "
                         "(ou --no-eval pour entraîner seulement)")
    if not args.dry and not _bootstrap.have_torch():
        raise SystemExit(_bootstrap.TORCH_HELP)
    pre = R.preflight(plan, args.data)
    for w in pre["avertissements"]:
        print("avertissement :", w)
    if pre["manquants"]:
        raise SystemExit("données manquantes pour ce plan (aucun entraînement lancé) :\n  " + "\n  ".join(pre["manquants"]))
    args.runs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    state = {}
    for i, r in enumerate(plan, 1):
        d = args.runs / r.run_id
        st = status_of(d)
        state[r.run_id] = st
        print(f"[{i:>2}/{len(plan)}] {r.run_id:<34} {st}")
    if args.dry:
        return

    common = ["--data", args.data, "--runs", args.runs, "--device", args.device]
    for i, r in enumerate(plan, 1):
        d = args.runs / r.run_id
        if status_of(d) == "évalué":
            continue
        if args.max_hours is not None and (time.time() - t0) / 3600 >= args.max_hours:
            print(f"budget de {args.max_hours} h atteint : arrêt propre (relancer pour continuer)")
            break
        remaining = None if args.max_hours is None else max(60.0, args.max_hours * 3600 - (time.time() - t0))
        print(f"\n=== [{i}/{len(plan)}] {r.run_id}", flush=True)
        if status_of(d) != "entraîné":
            targs = ["--run-id", r.run_id, "--steps", args.steps, "--batch-size", args.batch_size, "--lr", args.lr, *common]
            if remaining is not None:
                targs += ["--max-seconds", remaining]
            rc = call("train.py", *targs)
            if rc != 0:
                state[r.run_id] = f"ÉCHEC entraînement (code {rc})"
                print(state[r.run_id])
                _write_status(args.runs, state)
                continue
        if status_of(d) == "entraîné" and not args.no_eval:
            rc = call("evaluate.py", "--run-id", r.run_id, "--instrument", args.instrument, *common)
            if rc != 0:
                state[r.run_id] = f"ÉCHEC évaluation (code {rc})"
                print(state[r.run_id])
        state[r.run_id] = status_of(d) if "ÉCHEC" not in state[r.run_id] else state[r.run_id]
        _write_status(args.runs, state)
    _write_status(args.runs, state)
    done = sum(1 for v in state.values() if v == "évalué")
    print(f"\n{done}/{len(plan)} runs évalués en {(time.time() - t0) / 3600:.2f} h ; état dans {args.runs}/plan_status.json")
    print("Ensuite : uv run scripts/analyze.py")


def _write_status(runs: Path, state: dict) -> None:
    (runs / "plan_status.json").write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
