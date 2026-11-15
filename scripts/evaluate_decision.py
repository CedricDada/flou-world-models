#!/usr/bin/env python
"""Extension « décision » : mesure des runs ENTRAÎNÉS (niveau 1 : H4 ; niveau 2 : H5). PyTorch requis.

NON EXÉCUTÉ dans l'environnement de rédaction (pas de PyTorch) : `uv run scripts/smoke_test.py` fait travailler ce
code sur de minuscules données ; le lancer d'abord. Hypothèses et articles : docs/EXTENSION_DECISION.md.

    niveau 1   le modèle prédit k images, on y lit le contact, on compare au coût exact   -> runs/<run>/decision.json
               (rapide : quelques milliers d'états ; régression, diffusion, jetons du cœur)
    niveau 2   un planificateur joue dans le vrai jeu avec le modèle comme simulateur       -> runs/<run>/planning.json
               (COÛTEUX : games x 100 décisions x candidates x samples x horizon images, soit ~10 millions avec les
               réglages par défaut ; régression et diffusion du cœur seulement, sauf --families)

Exemples
    uv run scripts/validate_decision.py                                   # d'abord : contrôles H0e, H0f (sans GPU)
    uv run scripts/evaluate_decision.py --all                             # niveau 1 sur les runs du cœur
    uv run scripts/evaluate_decision.py --all --level 2 --games 4 --game-steps 10 --no-save   # chronométrer le niveau 2
    uv run scripts/evaluate_decision.py --all --level 2                   # niveau 2 complet (régression, diffusion)
    uv run scripts/evaluate_decision.py --run-id core-diffusion-p0.30-s0 --level 1 2

Pendant le niveau 2, la durée restante estimée s'affiche dès la première décision : interrompre (Ctrl-C) et réduire
--games ou --candidates si elle est trop longue. Changer ces réglages pour une partie des runs seulement fausserait
la comparaison : les garder identiques pour tous.
"""
import _bootstrap

import argparse
from pathlib import Path

from validate_decision import add_plan_args, plan_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--level", type=int, nargs="+", choices=[1, 2], default=[1])
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--groups", nargs="+", default=["core"], help="avec --all : groupes du plan concernés")
    ap.add_argument("--families", nargs="+", default=None,
                    help="avec --all (défaut : niveau 1 = regression diffusion tokens ; niveau 2 = regression diffusion)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5], help="niveau 1 : horizons (1 à 5)")
    ap.add_argument("--m", type=int, default=8, help="niveau 1 : tirages par état (générateurs)")
    ap.add_argument("--n-list", type=int, nargs="+", default=[1, 3], help="niveau 1 : pas d'Euler (diffusion)")
    ap.add_argument("--n-steps", type=int, default=3, help="niveau 2 : pas d'Euler (diffusion)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="niveau 2 : ne rien écrire (pour chronométrer)")
    ap.add_argument("--seed", type=int, default=0)
    add_plan_args(ap)
    args = ap.parse_args()

    _bootstrap.require_torch()                      # message clair si PyTorch manque
    from flou.train.evaluation import evaluate_decision_run, plan_run
    from flou.train.runs import RunSpec

    if args.run_id:
        todo = [args.run_id]
    else:
        todo = sorted(d.name for d in args.runs.iterdir() if (d / "final.pt").exists()) if args.runs.is_dir() else []
    if not todo:
        raise SystemExit(f"aucun run terminé dans {args.runs}/ (final.pt manquant) : lancer scripts/train.py")

    default_fams = {1: ("regression", "diffusion", "tokens"), 2: ("regression", "diffusion")}
    for level in sorted(set(args.level)):
        fams = tuple(args.families) if args.families else default_fams[level]
        out_name = "decision.json" if level == 1 else "planning.json"
        for rid in todo:
            d = args.runs / rid
            if not (d / "final.pt").exists():
                print(f"{rid} : pas de final.pt (entraînement non terminé), ignoré")
                continue
            spec = RunSpec.load(d)
            if spec.family == "oracle":
                continue                                    # le régresseur oracle reçoit la branche : hors sujet
            if args.all and (spec.group not in args.groups or spec.family not in fams):
                continue
            if (d / out_name).exists() and not args.force:
                print(f"{rid} : {out_name} existe déjà (--force pour refaire)")
                continue
            print(f"== niveau {level} : {rid}", flush=True)
            try:
                if level == 1:
                    evaluate_decision_run(d, args.data, device=args.device, ks=tuple(args.ks), m=args.m,
                                          n_list=tuple(args.n_list), seed=args.seed, amp=not args.no_amp)
                else:
                    plan_run(d, args.data, device=args.device, pc=plan_config(args), n_games=args.games,
                             seed=args.seed, n_steps=args.n_steps, game_steps=args.game_steps, amp=not args.no_amp,
                             save=not args.no_save)
            except FileNotFoundError as e:
                print(f"{rid} : ignoré ({e})")


if __name__ == "__main__":
    main()
