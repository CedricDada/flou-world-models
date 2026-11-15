#!/usr/bin/env python
"""Bandes d'images : ce que le modèle IMAGINE sur k pas, en regard de la réalité. PyTorch requis.

À lancer APRÈS un entraînement, sur un run terminé. Aucun ré-entraînement, aucune mesure : ce script ne
produit que des illustrations (rapport, soutenance). Les verdicts restent ceux de scripts/analyze.py.

Chaque état de bifurcation donne deux lignes : la vérité terrain, puis les images que le modèle prédit en
autorégressif (il se nourrit de ses propres sorties). C'est là que le flou se voit : la régression se
brouille pas après pas, un échantillonneur reste net mais peut dériver.

    runs/<run>/filmstrip_<variante>.png     la planche (2 lignes par état)
    runs/<run>/filmstrip_<variante>_<i>.gif un GIF par état, avec --gif (réel | prédit, côte à côte)

Exemples
    uv run scripts/filmstrip.py --run-id core-regression-p0.30-s0
    uv run scripts/filmstrip.py --run-id core-diffusion-p0.30-s0 --n-steps 3 --gif
    uv run scripts/filmstrip.py --all --states 3 --horizon 12
"""
import _bootstrap

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id")
    g.add_argument("--all", action="store_true", help="tous les runs terminés de --runs")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--states", type=int, default=4, help="nombre d'états de bifurcation illustrés")
    ap.add_argument("--horizon", type=int, default=16, help="pas imaginés (borné par le jeu de test)")
    ap.add_argument("--n-steps", type=int, default=3, help="pas d'Euler (diffusion)")
    ap.add_argument("--tau", type=float, default=1.0, help="température (jetons)")
    ap.add_argument("--scale", type=int, default=3, help="agrandissement des vignettes 64x64")
    ap.add_argument("--gif", action="store_true", help="écrit aussi un GIF par état")
    ap.add_argument("--gif-ms", type=int, default=400, help="durée d'une image du GIF, en ms")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _bootstrap.require_torch()                      # message clair si PyTorch manque
    from flou.data.dataset import BifurcationSet
    from flou.eval.rollout import autoregressive_rollout, rollout_truth
    from flou.game.viz import contact_sheet
    from flou.train.predictors import TorchPredictor, load_run
    from flou.train.runs import RunSpec, test_file

    if args.run_id:
        todo = [args.run_id]
    else:
        todo = sorted(d.name for d in args.runs.iterdir() if (d / "final.pt").exists()) if args.runs.is_dir() else []
    if not todo:
        raise SystemExit(f"aucun run terminé dans {args.runs}/ (final.pt manquant) : lancer scripts/train.py")

    for rid in todo:
        d = args.runs / rid
        if not (d / "final.pt").exists():
            print(f"{rid} : pas de final.pt, ignoré")
            continue
        spec = RunSpec.load(d)
        if spec.family == "oracle":
            print(f"{rid} : oracle (reçoit la branche), hors sujet pour une illustration")
            continue
        _, model = load_run(d, args.device)

        bset = BifurcationSet.load(test_file(args.data, spec.p))
        horizon = max(1, min(args.horizon, bset.horizon))
        sub = bset.subset(np.arange(min(args.states, len(bset))))

        kw = {}
        variante = "net"
        if spec.family == "diffusion":
            kw, variante = {"n_steps": args.n_steps}, f"n{args.n_steps}"
        elif spec.family == "tokens":
            kw, variante = {"tau": args.tau}, f"tau{args.tau:g}"
        pred = TorchPredictor(model, args.device, name=rid, amp=not args.no_amp, **kw)

        vrai, _ = rollout_truth(sub, horizon, seed=args.seed)                 # (n, H, 64, 64, 3) uint8
        imag = autoregressive_rollout(pred, sub, horizon, seed=args.seed)     # (n, H, 64, 64, 3) uint8

        rows, labels = [], []
        for i in range(len(sub)):
            rows += [list(vrai[i]), list(imag[i])]
            labels += [[f"réel t+{h + 1}" for h in range(horizon)],
                       [f"{spec.family[:4]} t+{h + 1}" for h in range(horizon)]]
        path = d / f"filmstrip_{variante}.png"
        contact_sheet(rows, path, scale=args.scale, labels=labels)
        print(f"{rid} : {path}  ({len(sub)} états x {horizon} pas)")

        if args.gif:
            from flou.game.viz import _pil
            Image, _ = _pil()
            for i in range(len(sub)):
                frames = []
                for h in range(horizon):
                    pair = np.concatenate([vrai[i, h], np.full((64, 4, 3), 40, np.uint8), imag[i, h]], axis=1)
                    k = args.scale * 2
                    frames.append(Image.fromarray(pair).resize((pair.shape[1] * k, 64 * k), Image.NEAREST))
                gif = d / f"filmstrip_{variante}_{i}.gif"
                frames[0].save(gif, save_all=True, append_images=frames[1:], duration=args.gif_ms, loop=0)
                print(f"   {gif}  (réel | imaginé)")


if __name__ == "__main__":
    main()
