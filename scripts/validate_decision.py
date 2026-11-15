#!/usr/bin/env python
"""Extension « décision » : valide les instruments AVANT de mesurer un modèle (contrôles H0e et H0f). Sans GPU.

Ce que fait le script (voir docs/EXTENSION_DECISION.md) :
  1. crée, s'il manque, le jeu de test du niveau 1 : data/p0.30/test_decision.npz (3 000 états de bifurcation,
     flux aléatoire distinct du jeu de test principal) ;
  2. H0e : le lecteur de contact retrouve-t-il l'événement du jeu sur de vraies images ? (accord >= 99 %) ;
  3. table « attendu si les modèles étaient parfaits » (prédicteurs factices), comparée à la théorie à k = 1 ;
  4. H0f : le planificateur fonctionne-t-il avec le VRAI jeu comme modèle ? (~10 min sur 2 cœurs avec les réglages
     par défaut : 50 parties de 100 pas ; --no-planner pour sauter cette étape, --games pour la raccourcir).

Exemples
    uv run scripts/validate_decision.py
    uv run scripts/validate_decision.py --no-planner            # H0e et table seulement (quelques secondes)
    uv run scripts/validate_decision.py --games 16              # H0f plus court (moins précis)
Sortie : reports/decision_instrument.json (lu par scripts/analyze.py).
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from flou.data.dataset import BifurcationSet, build_bifurcation_set
from flou.eval import analysis as A
from flou.eval import decision_instrument as DI
from flou.eval import planning as P
from flou.game import GameConfig

DECISION_SEED_OFFSET = 1000      # flux aléatoire distinct du jeu de test principal (graine + 1000)


def add_plan_args(ap: argparse.ArgumentParser) -> None:
    d = P.PlanConfig()
    ap.add_argument("--games", type=int, default=50, help="parties jouées par le planificateur")
    ap.add_argument("--candidates", type=int, default=d.candidates, help="J : suites d'actions tirées par décision")
    ap.add_argument("--horizon", type=int, default=d.horizon, help="H : pas imaginés")
    ap.add_argument("--samples", type=int, default=d.samples, help="m : tirages du modèle par suite (générateurs)")
    ap.add_argument("--lam", type=float, default=d.lam, help="poids du terme de rapprochement du petit objet")
    ap.add_argument("--reader", choices=["soft", "hard"], default=d.reader, help="lecteur de contact")
    ap.add_argument("--game-steps", type=int, default=None, help="pas joués par partie (défaut : partie entière)")


def plan_config(args) -> P.PlanConfig:
    return P.PlanConfig(candidates=args.candidates, horizon=args.horizon, samples=args.samples, lam=args.lam,
                        reader=args.reader)


def ensure_decision_set(data: Path, p: float, n: int, seed: int, config: str | None = None) -> BifurcationSet:
    path = data / f"p{p:.2f}" / "test_decision.npz"
    if path.exists():
        return BifurcationSet.load(path)
    base = GameConfig.from_json(config) if config else GameConfig()
    cfg = base.with_p(p)
    bs = build_bifurcation_set(cfg, n=n, seed=seed + DECISION_SEED_OFFSET)
    bs.save(path, {"config": cfg.to_dict(), "split": "test-decision", "seed": seed + DECISION_SEED_OFFSET, "n": n})
    print(f"créé : {path} ({n} états de bifurcation)")
    return bs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--states", type=int, default=3000, help="taille du jeu de test du niveau 1 (s'il faut le créer)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5], help="horizons (1 à 5)")
    ap.add_argument("--m", type=int, default=8, help="tirages par état pour les générateurs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=str, default=None, help="JSON de GameConfig (facultatif)")
    ap.add_argument("--no-planner", action="store_true", help="ne pas lancer le contrôle H0f (long)")
    ap.add_argument("--out", type=Path, default=Path("reports/decision_instrument.json"))
    add_plan_args(ap)
    args = ap.parse_args()

    bset = ensure_decision_set(args.data, args.p, args.states, args.seed, args.config)

    def progress(t, total, sec):
        if t in (1, 5) or t % 20 == 0:
            print(f"   H0f : décision {t}/{total}, {sec:.0f} s écoulées, fin estimée dans "
                  f"{sec / t * (total - t) / 60:.1f} min", flush=True)

    rep = DI.validate_decision(bset, tuple(args.ks), args.m, args.seed, planner=not args.no_planner,
                               pc=plan_config(args), n_games=args.games, game_steps=args.game_steps, progress=progress)
    path = DI.save_report(rep, args.out)

    h = rep["h0e"]
    print(f"{rep['n_states']} états de bifurcation (p = {rep['p']}) ; part à risque ambigu : "
          + ", ".join(f"{k} {100 * v:.1f} %" for k, v in rep["ambiguous_share"].items()))
    print(f"H0e (lecteur de contact sur de vraies images) : accord {100 * h['agreement']:.2f} % sur {h['n_images']} "
          f"images dont {h['n_contacts']} contacts -> {'réussi' if h['pass'] else 'ÉCHOUÉ'}")
    print("\nErreur |Ĉ - C*| attendue si les modèles étaient parfaits (états à risque ambigu) :")
    print(A.markdown_table(rep["expected"]))
    f = rep["h0f"]
    if f:
        for x in (f["random"], f["expert"], f["planner"]):
            print(P.play_line(x))
        print(f"H0f (planificateur avec le vrai jeu) : {f['planner']['contacts']:.2f} contact par partie (max "
              f"{f['max_contacts']}), +{f['gain_over_random']:.2f} points sur le bot aléatoire (min {f['min_gain']}) "
              f"-> {'réussi' if f['pass'] else 'ÉCHOUÉ'}")
    else:
        print("H0f : non lancé (--no-planner) ; il reste « non évaluable » dans l'analyse.")
    print(f"rapport : {path}")
    if not h["pass"] or (f and not f["pass"]):
        raise SystemExit("contrôle ÉCHOUÉ : ne mesurer aucun modèle avec l'extension avant de comprendre pourquoi.")


if __name__ == "__main__":
    main()
