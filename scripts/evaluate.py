#!/usr/bin/env python
"""Évalue un ou plusieurs runs entraînés sur le jeu de test de bifurcation (écrit eval.json et eval_arrays.npz).

NON EXÉCUTÉ dans l'environnement de rédaction (pas de PyTorch) : lancer d'abord `uv run scripts/smoke_test.py`.

Exemples
    uv run scripts/evaluate.py --run-id core-regression-p0.30-s0
    uv run scripts/evaluate.py --all                         # tous les runs terminés et pas encore évalués
    uv run scripts/evaluate.py --all --force                 # tout réévaluer (par exemple après un autre ρmax)
    uv run scripts/evaluate.py --run-id core-diffusion-p0.30-s0 --n-samples 3 --rollout-n 100

ρmax est lu dans reports/instrument.json (uv run scripts/validate_instrument.py d'abord) ; --rho-max le remplace.
La sonde de position des rollouts est analytique (référence) ; --probe-cnn donne une sonde apprise
(uv run scripts/train_probe.py) : la comparer d'abord à l'analytique avant de s'y fier.
"""
import _bootstrap

import argparse
from pathlib import Path

from flou.eval import instrument


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--instrument", type=Path, default=Path("reports/instrument.json"))
    ap.add_argument("--rho-max", type=float, default=None)
    ap.add_argument("--probe-cnn", type=Path, default=None, help="reports/probe.pt (sonde apprise, facultative)")
    ap.add_argument("--n-list", type=int, nargs="+", default=[1, 3, 5, 20], help="pas d'Euler (diffusion)")
    ap.add_argument("--n-samples", type=int, default=1, help="tirages par état de test (1 000 états x n)")
    ap.add_argument("--rollout-n", type=int, default=250, help="états utilisés pour les rollouts de 16 pas (0 : aucun)")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.rho_max is not None:
        rho = args.rho_max
    elif args.instrument.exists():
        rho = instrument.load_rho_max(args.instrument)
    else:
        raise SystemExit(f"{args.instrument} absent : lancer uv run scripts/validate_instrument.py (ou donner --rho-max)")

    torch = _bootstrap.require_torch()                      # message clair si PyTorch manque
    from flou.train.evaluation import evaluate_run

    probe = None
    if args.probe_cnn:
        from flou.models.probe_cnn import PositionProbe, ProbeAdapter
        net = PositionProbe()
        net.load_state_dict(torch.load(args.probe_cnn, map_location=args.device, weights_only=False)["model"])
        probe = ProbeAdapter(net, args.device)

    if args.run_id:
        todo = [args.run_id]
    else:
        todo = sorted(d.name for d in args.runs.iterdir() if (d / "final.pt").exists()) if args.runs.is_dir() else []
    if not todo:
        raise SystemExit(f"aucun run terminé dans {args.runs}/ (final.pt manquant) : lancer scripts/train.py")
    for rid in todo:
        d = args.runs / rid
        if not (d / "final.pt").exists():
            print(f"{rid} : pas de final.pt (entraînement non terminé), ignoré")
            continue
        if (d / "eval.json").exists() and not args.force:
            print(f"{rid} : déjà évalué (--force pour refaire)")
            continue
        print(f"== {rid} (ρmax = {rho:.4f})")
        evaluate_run(d, args.data, rho, device=args.device, n_list=tuple(args.n_list), n_samples=args.n_samples,
                     rollout_n=args.rollout_n, horizon=args.horizon, probe=probe, seed=args.seed, amp=not args.no_amp)


if __name__ == "__main__":
    main()
