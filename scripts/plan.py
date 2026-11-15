#!/usr/bin/env python
"""Affiche le plan de 25 entraînements, dans l'ordre de priorité, avec budget et estimation de durée.

Exemples
    uv run scripts/plan.py                                  # plan par défaut (20 000 pas, lot de 64)
    uv run scripts/plan.py --steps 8000                     # plan réduit (T4 gratuite, tests)
    uv run scripts/plan.py --timing reports/timing.json     # durée estimée d'après les temps mesurés
    uv run scripts/plan.py --budget-hours 38 --timing reports/timing.json   # nombre de pas qui tient dans 38 h

Sans `--timing`, aucune durée n'est donnée : le temps par pas dépend du GPU et n'a pas été mesuré ici
(`scripts/smoke_test.py --time` le mesure).
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from flou.models.params import human
from flou.models.registry import expected_params
from flou.train import runs as R


def hours_for(plan, timing):
    return R.estimate_hours(plan, timing)


def steps_for_budget(steps: int, batch: int, budget_h: float, timing: dict, **kw) -> int:
    """Plus grand nombre de pas (arrondi à 100) dont le plan tient dans le budget. Le temps est linéaire en pas."""
    per_step = hours_for(R.build_plan(steps=1, batch_size=batch, **kw), timing)
    return max(100, int(budget_h / per_step + 1e-6) // 100 * 100)      # + 1e-6 : 14399,999999 doit donner 14400


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--timing", type=Path, default=None, help="reports/timing.json (secondes par pas, mesurées)")
    ap.add_argument("--budget-hours", type=float, default=None, help="avec --timing : calcule le nombre de pas")
    ap.add_argument("--json", action="store_true", help="sortie JSON (une entrée par run)")
    args = ap.parse_args()

    timing = json.loads(args.timing.read_text(encoding="utf-8"))["seconds_per_step"] if args.timing else None
    steps = args.steps
    if args.budget_hours is not None:
        if timing is None:
            raise SystemExit("--budget-hours demande --timing (le temps par pas doit être mesuré)")
        steps = steps_for_budget(steps, args.batch_size, args.budget_hours, timing, lr=args.lr)
        print(f"budget {args.budget_hours:g} h -> {steps} pas par run\n")
    plan = R.build_plan(steps=steps, batch_size=args.batch_size, lr=args.lr)

    if args.json:
        print(json.dumps([dict(r.to_dict(), run_id=r.run_id, images_seen=r.images_seen,
                               params=expected_params(r.model_spec())) for r in plan], indent=1))
        return
    print(f"{'#':>2}  {'run':<34} {'groupe':<8} {'p':>4} {'graine':>6} {'params':>8} {'pas':>8} {'images vues':>12}  note")
    for i, r in enumerate(plan, 1):
        print(f"{i:>2}  {r.run_id:<34} {r.group:<8} {r.p:>4.2f} {r.seed:>6} {human(expected_params(r.model_spec())):>8} "
              f"{r.steps:>8} {r.images_seen:>12,}  {r.note}".replace(",", " "))
    s = R.summarize_plan(plan)
    print("\n" + ", ".join(f"{g} {s[g]}" for g in R.GROUPS) + f"  =  {s['total']} runs")
    print("Budget apparié : mêmes pas x lot (images vues) pour toutes les familles ; pour la famille C, le VQ-VAE et le "
          f"Transformer se partagent ce total ({R.RunSpec('tokens', 'core', .3, 0, K=256).vq_fraction:.0%} / reste).")
    if timing:
        h = hours_for(plan, timing)
        print(f"Durée estimée d'après {args.timing} : {h:.1f} h de GPU au total, {h / s['total']:.2f} h par run en moyenne.")
        core = [r for r in plan if r.group in ("core", "control")]
        print(f"  dont cœur + contrôles (jamais sacrifiés) : {hours_for(core, timing):.1f} h")
    else:
        print("Durée : non estimée (aucun temps par pas mesuré ; lancer scripts/smoke_test.py --time sur le GPU visé).")


if __name__ == "__main__":
    main()
