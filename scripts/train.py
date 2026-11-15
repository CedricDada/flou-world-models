#!/usr/bin/env python
"""Entraîne UN run du plan (reprenable : relancer la même commande repart du dernier checkpoint).

NON EXÉCUTÉ dans l'environnement de rédaction (pas de PyTorch) : lancer d'abord `uv run scripts/smoke_test.py`.

Exemples
    uv run scripts/train.py --run-id core-regression-p0.30-s0
    uv run scripts/train.py --run-id core-diffusion-p0.30-s1 --steps 8000      # budget réduit
    uv run scripts/train.py --run-id core-tokens-K256-p0.30-s0 --max-seconds 3000   # s'arrête proprement, reprend ensuite
    uv run scripts/plan.py                                                       # liste des identifiants

Prérequis : `uv run scripts/generate_data.py --split train --p 0.3` et `--split val --p 0.3` (et les autres p du plan).
Les paramètres d'un run déjà commencé sont relus dans runs/<run-id>/spec.json (les options --steps etc. sont alors
ignorées, avec un avertissement) : on ne change pas un budget en cours d'entraînement.
"""
import _bootstrap

import argparse
import json
from pathlib import Path

from flou.train import runs as R


def find_spec(run_id: str, steps: int, batch_size: int, lr: float) -> R.RunSpec:
    plan = {r.run_id: r for r in R.build_plan(steps=steps, batch_size=batch_size, lr=lr)}
    if run_id not in plan:
        raise SystemExit(f"run inconnu : {run_id}\nidentifiants valides :\n  " + "\n  ".join(plan))
    return plan[run_id]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seconds", type=float, default=None, help="arrêt propre après ce temps ; relancer pour reprendre")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true", help="désactive la précision mixte fp16")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-n", type=int, default=2048, help="transitions de validation")
    args = ap.parse_args()

    run_dir = args.runs / args.run_id
    spec = find_spec(args.run_id, args.steps, args.batch_size, args.lr)
    if (run_dir / "spec.json").exists():
        saved = R.RunSpec.load(run_dir)
        if saved != spec:
            print(f"avertissement : {run_dir}/spec.json diffère des options ; on garde le spec sauvegardé "
                  f"(pas {saved.steps}, lot {saved.batch_size})")
        spec = saved
    if (run_dir / "final.pt").exists():
        print(f"{args.run_id} : déjà terminé (final.pt présent)")
        return
    torch = _bootstrap.require_torch()                      # message clair si PyTorch manque
    from flou.train.frames import FrameBank
    from flou.train.tasks import build_task
    from flou.train.trainer import Trainer, TrainConfig

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA indisponible : passer --device cpu (très lent) ou changer d'environnement (Colab : GPU T4)")

    spec.save(run_dir)
    print(f"{spec.run_id} : {spec.steps} pas, lot {spec.batch_size}, p = {spec.p}, graine {spec.seed}")
    bank = FrameBank(R.train_files(args.data, spec.p, "train"), spec.k_ctx, args.device)
    try:
        val_bank = FrameBank(R.train_files(args.data, spec.p, "val"), spec.k_ctx, args.device)
    except FileNotFoundError as e:
        print(f"avertissement : pas de validation ({e})")
        val_bank = None
    cfg = TrainConfig(batch_size=spec.batch_size, lr=spec.lr, ema=spec.ema, seed=spec.seed, amp=not args.no_amp,
                      ckpt_every=args.ckpt_every, val_every=args.val_every)
    model, stages = build_task(spec, bank, val_bank, cfg, args.device, val_n=args.val_n)
    tr = Trainer(run_dir, model, stages, cfg, args.device, meta={"run_id": spec.run_id, "family": spec.family,
                                                                  "group": spec.group, "p": spec.p, "seed": spec.seed})
    out = tr.run(max_seconds=args.max_seconds)
    if out["finished"]:
        info = json.loads((run_dir / "compute.json").read_text(encoding="utf-8"))
        print(f"terminé : {info['gpu_hours']:.2f} h sur {info['gpu']}, {info['parameters']:,} paramètres".replace(",", " "))
    else:
        print(f"interrompu après {out['train_seconds'] / 3600:.2f} h : relancer la même commande pour reprendre")


if __name__ == "__main__":
    main()
