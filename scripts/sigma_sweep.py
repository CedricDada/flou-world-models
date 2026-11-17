#!/usr/bin/env python
"""Diagnostic EXPLORATOIRE : un pas d'Euler depuis différents σ de départ (diffusion seulement).

Déclaré dans `docs/DECISIONS.md` (révision du 17 novembre 2026) à la suite de l'incident J9.
**Aucun verdict n'en dépend** : H2a reste évaluée telle que préenregistrée, sur n = 1 contre n = 3.

La question. Le cours attend d'un échantillon en un pas qu'il soit D(x ; σmax), c'est-à-dire la
moyenne conditionnelle : une image floue, α̂ ≈ p, ρ ≈ 0, MSE ≈ 1 fois le plancher de Bayes. Mesuré,
on obtient tout autre chose : 100 % d'images « autre » et une MSE à 2,1 fois le plancher, sur les
deux graines. Explication proposée : σmax = 80 n'a presque jamais été vu à l'entraînement — σ y suit
une loi log-normale (P_mean = −1,2, P_std = 1,2), ce qui donne 4 exemples au-dessus de 80 et 525
au-dessus de 20, sur 2 227 200.

COMMENT LIRE LE RÉSULTAT. Pas contre « flou élevé à tout σ », qui est faux même pour un débruiteur
EXACT. Le pas part de x = σ·n, du bruit pur ; l'a posteriori sur {A, B} a un logit d'écart-type
‖A − B‖/σ ≈ 15/σ. Dès que σ < ‖A − B‖, le bruit seul tranche la branche et l'image sort NETTE (A ou
B), avec ρ = 0. Le flou n'apparaît qu'aux grands σ. La référence, à comparer ligne à ligne :

        σ     1     2     5    10    20    30    40    60    80
    floue   7 %  14 %  34 %  56 %  76 %  86 %  92 %  98 %  99,8 %
    α̂ p5   0,00  0,00 0,003 0,034 0,109 0,158 0,187 0,221  0,239
    α̂ p95  1,00  1,00 0,985 0,842 0,600 0,496 0,444 0,394  0,370

(`uv run scripts/ideal_sigma_reference.py`, NumPy seul ; « autre » vaut 0 à tout σ par construction.) Ce
qui sépare le réseau de cette référence est donc le grain (ρ élevé) et la dispersion de α̂.

PRUDENCE SUR LES σ PETITS. Le réseau a été entraîné sur « image + σ·bruit », et les images cibles ont
une moyenne de −0,77 par pixel. Un bruit pur à σ = 1 s'en écarte de 85 écarts-types sur la moyenne,
à σ = 10 de 8,5, à σ = 20 de 4, à σ = 80 de 1. Le point de départ n'est dans la loi d'entraînement
qu'à σ ≳ 40, là où il y a 51 exemples (σ ≥ 40) ou 4 (σ ≥ 80), contre 21 400 à σ ≥ 5. Un mauvais
résultat à σ ≤ 10 ne dirait donc pas « σ mal couvert » : il peut venir du point de départ.

Observation à regarder, sans en faire une conclusion : le α̂ mesuré à n = 1 va de 0,03 à 0,86
(5e–95e centiles, J9), ce que le débruiteur exact donne à σ ≈ 10 (0,034–0,842) et non à σ = 80.

Écrit `runs/<run>/sigma_sweep.json`. Sans effet sur `eval.json`. **Idempotent** : un run qui a déjà son
`sigma_sweep.json` est sauté, sauf avec `--force` (les deux interrupteurs du notebook peuvent donc rester
allumés d'une session à l'autre).

Exemples
    uv run scripts/sigma_sweep.py --run-id core-diffusion-p0.30-s0
    uv run scripts/sigma_sweep.py --all --sigmas 1 2 5 10 20 30 40 60 80
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
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0])
    ap.add_argument("--n-states", type=int, default=0, help="0 : tout le jeu de test ; sinon les N premiers états")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="refaire un balayage déjà écrit")
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
        sortie = d / "sigma_sweep.json"
        if sortie.exists() and not args.force:
            print(f"{rid} : balayage déjà fait ({sortie.name}), sauté (--force pour refaire)")
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
        sortie.write_text(json.dumps({"run_id": rid, "p": spec.p, "rho_max": rho_max,
                                      "n_states": len(bset), "exploratoire": True,
                                      "lecture": "un pas d'Euler depuis chaque sigma ; aucun verdict n'en depend",
                                      "rows": lignes}, indent=1), encoding="utf-8")
        print(f"   -> {sortie}")


if __name__ == "__main__":
    main()
