#!/usr/bin/env python
"""Diagnostic EXPLORATOIRE : un pas d'Euler depuis différents σ de départ (diffusion seulement).

Déclaré dans `docs/DECISIONS.md` (révision du 17 novembre 2026) à la suite de l'incident J9.
**Aucun verdict n'en dépend** : H2a reste évaluée telle que préenregistrée, sur n = 1 contre n = 3.

La question. Le cours attend d'un échantillon en un pas qu'il soit D(x ; σmax), c'est-à-dire la
moyenne conditionnelle : une image floue, α̂ ≈ p, ρ ≈ 0, MSE ≈ 1 fois le plancher de Bayes. Mesuré,
on obtient tout autre chose : 100 % d'images « autre » et une MSE à 2,1 fois le plancher, sur les
deux graines. L'explication proposée est que σmax = 80 n'a presque jamais été vu à l'entraînement —
σ y suit une loi log-normale (P_mean = −1,2, P_std = 1,2), ce qui donne 4 exemples au-dessus de 80 et
525 au-dessus de 20, sur 2 227 200. Le préconditionnement d'EDM va dans le même sens : à σ = 80,
c_in ≈ 1/80 et c_skip ≈ 4·10⁻⁵, donc le réseau ne reçoit presque que du bruit.

Ce script teste cette explication sans réentraîner : si elle est juste, un pas depuis un σ **bien
couvert** à l'entraînement (1 à 5) doit redonner le comportement attendu — flou élevé, α̂ resserré
autour de p, ρ petit — et la dégradation doit apparaître progressivement en montant vers 80.

Écrit `runs/<run>/sigma_sweep.json`. Sans effet sur `eval.json`.

Exemples
    uv run scripts/sigma_sweep.py --run-id core-diffusion-p0.30-s0
    uv run scripts/sigma_sweep.py --all --sigmas 1 2 5 10 20 80
"""
import _bootstrap

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flou.eval import instrument


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id")
    g.add_argument("--all", action="store_true", help="tous les runs de diffusion terminés")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--instrument", type=Path, default=Path("reports/instrument.json"))
    ap.add_argument("--rho-max", type=float, default=None)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0, 20.0, 80.0])
    ap.add_argument("--n-states", type=int, default=0, help="0 : tout le jeu de test ; sinon les N premiers états")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.rho_max is not None:
        rho_max = args.rho_max
    elif args.instrument.exists():
        rho_max = instrument.load_rho_max(args.instrument)
    else:
        raise SystemExit(f"{args.instrument} absent : lancer uv run scripts/validate_instrument.py (ou --rho-max)")

    _bootstrap.require_torch()
    from flou.data.dataset import BifurcationSet
    from flou.eval import evaluate as ev
    from flou.train.predictors import TorchPredictor, load_run
    from flou.train.runs import test_file

    if args.run_id:
        todo = [args.run_id]
    else:
        todo = sorted(d.name for d in args.runs.iterdir() if (d / "final.pt").exists()) if args.runs.is_dir() else []
    if not todo:
        raise SystemExit(f"aucun run terminé dans {args.runs}/ (final.pt manquant)")

    for rid in todo:
        d = args.runs / rid
        if not (d / "final.pt").exists():
            print(f"{rid} : pas de final.pt, ignoré")
            continue
        spec, model = load_run(d, args.device)
        if spec.family != "diffusion":
            if args.run_id:
                raise SystemExit(f"{rid} est de la famille « {spec.family} » : le balayage de σ ne concerne "
                                 "que la diffusion (le σ de départ n'existe que pour l'échantillonneur d'Euler)")
            continue
        bset = BifurcationSet.load(test_file(args.data, spec.p))
        if args.n_states:
            bset = bset.subset(np.arange(min(args.n_states, len(bset))))
        print(f"== {rid}  (p = {spec.p}, ρmax = {rho_max:.4f}, {len(bset)} états, un seul pas d'Euler)")
        print(f"{'sigma':>7} {'floue':>7} {'autre':>7} {'nettes':>7} {'alpha moyen':>12} "
              f"{'alpha p5-p95':>16} {'rho median':>11} {'MSE/plancher':>13}")
        lignes = []
        for sigma in args.sigmas:
            t0 = time.time()
            pred = TorchPredictor(model, args.device, name=f"{rid}[1 pas, sigma={sigma:g}]", n_steps=1,
                                  sigma_max=float(sigma), amp=not args.no_amp)
            res = ev.evaluate_predictor(pred, bset, rho_max, seed=args.seed, keep_images=False)
            c, n = res["counts"], res["n"]
            alpha = np.asarray(res["per_sample"]["alpha"], dtype=float)
            rho = np.asarray(res["per_sample"]["rho"], dtype=float)
            p5, p95 = (float(x) for x in np.nanpercentile(alpha, [5, 95]))
            ligne = {"sigma": float(sigma), "n": n,
                     "floue": c["floue"] / n, "autre": c["autre"] / n,
                     "nettes": (c["nette-A"] + c["nette-B"]) / n,
                     "alpha_mean": res["alpha_mean"], "alpha_p5": p5, "alpha_p95": p95,
                     "rho_median": float(np.nanmedian(rho)),
                     "ratio_expected": res["mse"]["ratio_expected"],
                     "p_hat": res["p_hat"], "counts": c, "seconds": time.time() - t0}
            lignes.append(ligne)
            print(f"{sigma:>7.3g} {ligne['floue']:>6.1%} {ligne['autre']:>6.1%} {ligne['nettes']:>6.1%}"
                  f" {ligne['alpha_mean']:>12.3f} {p5:>7.3f}-{p95:<8.3f} {ligne['rho_median']:>11.3f}"
                  f" {ligne['ratio_expected']:>13.2f}")
        sortie = d / "sigma_sweep.json"
        sortie.write_text(json.dumps({"run_id": rid, "p": spec.p, "rho_max": rho_max,
                                      "n_states": len(bset), "exploratoire": True,
                                      "lecture": "un pas d'Euler depuis chaque sigma ; aucun verdict n'en depend",
                                      "rows": lignes}, indent=1), encoding="utf-8")
        print(f"   -> {sortie}")


if __name__ == "__main__":
    main()
