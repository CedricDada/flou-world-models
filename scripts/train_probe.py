#!/usr/bin/env python
"""Entraîne la sonde de position apprise (CNN) et la valide contre la vérité terrain.

NON EXÉCUTÉ dans l'environnement de rédaction (pas de PyTorch).

La sonde de référence des rollouts est ANALYTIQUE (flou.eval.probe.AnalyticProbe, sans apprentissage) ; cette sonde
apprise est un recoupement : elle sert à vérifier que la mesure ne dépend pas d'un détail de la sonde analytique
et à mesurer des images moins parfaites que le rendu exact. Critère de validation : RMSE < 0,5 pixel sur des
images de validation réelles (position de l'agent, de l'obstacle, de l'ennemi et du petit objet).

Exemples
    uv run scripts/train_probe.py                       # 6 000 pas, p = 0,3 (les positions ne dépendent pas de p)
    uv run scripts/train_probe.py --steps 12000 --p 0.5
Sorties : runs_probe/final.pt, reports/probe.pt (copie), reports/probe.json.
"""
import _bootstrap

import argparse
import json
import shutil
from pathlib import Path

from flou.train import runs as R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--run-dir", type=Path, default=Path("runs_probe"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold-px", type=float, default=0.5)
    args = ap.parse_args()

    torch = _bootstrap.require_torch()                      # message clair si PyTorch manque
    from flou.train.frames import FrameBank
    from flou.train.tasks import build_probe_task
    from flou.train.trainer import Trainer, TrainConfig

    bank = FrameBank(R.train_files(args.data, args.p, "train"), 1, args.device, with_positions=True)
    val_bank = FrameBank(R.train_files(args.data, args.p, "val"), 1, args.device, with_positions=True)
    cfg = TrainConfig(batch_size=args.batch_size, lr=args.lr, warmup=200, val_every=1000, ckpt_every=1000, seed=0)
    model, stages = build_probe_task(args.steps, bank, val_bank, cfg, args.device)
    Trainer(args.run_dir, model, stages, cfg, args.device, meta={"run_id": "probe"}).run()

    model.eval()
    with torch.no_grad():
        rmse = stages[0].val_fn()["rmse_px"]
    ok = rmse < args.threshold_px
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.run_dir / "final.pt", args.out / "probe.pt")
    rep = {"rmse_px": rmse, "threshold_px": args.threshold_px, "pass": bool(ok), "steps": args.steps, "p": args.p}
    (args.out / "probe.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"RMSE de validation : {rmse:.3f} px (seuil {args.threshold_px}) -> {'réussi' if ok else 'ÉCHOUÉ'}")
    if not ok:
        raise SystemExit("sonde apprise non validée : ne pas l'utiliser (--probe-cnn) ; la sonde analytique reste la référence")


if __name__ == "__main__":
    main()
