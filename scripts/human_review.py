#!/usr/bin/env python
"""Relecture humaine : estimer les erreurs du classifieur (100 cas) et comparer deux modèles en aveugle (A/B).

Quatre sous-commandes
    review-make   prépare 100 cas stratifiés par classe automatique (planches PNG + review.csv à remplir)
    review-score  compare vos classes (colonne human_class) à la classe automatique
    ab-make       prépare 20 paires issues des mêmes contextes pour 3 relecteurs (ordre gauche/droite aléatoire)
    ab-score      dépouille les réponses des relecteurs avec la clé

Un « modèle » se désigne par `run_id` ou `run_id:variante` (variantes : n=1, n=3, n=5, n=20 pour la diffusion ;
tau=1 pour les jetons ; rien pour la régression et l'oracle).

Exemples
    uv run scripts/human_review.py review-make --select core-regression-p0.30-s0 core-diffusion-p0.30-s0:n=3 \\
        core-diffusion-p0.30-s0:n=1 core-tokens-K256-p0.30-s0:tau=1 --out reports/human/review
    uv run scripts/human_review.py review-score --dir reports/human/review

    uv run scripts/human_review.py ab-make --a core-diffusion-p0.30-s0:n=3 --b core-regression-p0.30-s0 \\
        --out reports/human/ab
    uv run scripts/human_review.py ab-score --dir reports/human/ab \\
        --responses reviewer_1.csv reviewer_2.csv reviewer_3.csv

Conseils : relire SANS regarder la classe automatique (elle n'est pas sur les planches) ; chaque relecteur remplit son
propre fichier ; ne remettre key.csv à personne avant la fin. Les évaluations des runs doivent exister (eval_arrays.npz).
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from flou.eval import human


def _write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def cmd_review_make(a) -> None:
    sel = [human.parse_selection(s) for s in a.select]
    pool = human.load_pool(a.runs, a.data, sel)
    if len(pool["labels"]) < a.n:
        print(f"attention : seulement {len(pool['labels'])} images disponibles pour {a.n} cas demandés")
    meta = human.write_review(a.out, pool["imgs"], pool["A"], pool["B"], pool["labels"], pool["alpha"], pool["rho"],
                              pool["source"], n=min(a.n, len(pool["labels"])), seed=a.seed, population=pool["population"])
    print(f"{meta['n_cases']} cas -> {a.out}/ (review_page_*.png, review.csv)")
    print("population poolée :", meta["population"])
    print("Remplir la colonne human_class de review.csv (A, B, floue, autre), puis : review-score --dir", a.out)


def cmd_review_score(a) -> None:
    s = human.score_review(Path(a.dir) / "review.csv")
    _write_json(s, Path(a.dir) / "review_score.json")
    if s["n_scored"] == 0:
        raise SystemExit(f"aucune ligne remplie dans {a.dir}/review.csv ({s['n_blank']} vides)")
    print(f"{s['n_scored']} cas notés ({s['n_blank']} vides)")
    ac = s["accord_brut"]
    print(f"accord brut classifieur/humain : {ac['value']:.2f}  IC95 [{ac['ci_lo']:.2f} ; {ac['ci_hi']:.2f}]   κ = {s['kappa_auto_humain']:.2f}")
    if "exactitude_ponderee_par_prevalence" in s:
        print(f"exactitude pondérée par la prévalence de la population : {s['exactitude_ponderee_par_prevalence']:.2f}")
    for name, v in s["accord_par_classe"].items():
        print(f"  {name:<8} {v['k']}/{v['n']} = {v['value']:.2f}  IC95 [{v['ci_lo']:.2f} ; {v['ci_hi']:.2f}]")
    print("lignes = classe automatique, colonnes = classe humaine", s["classes"])
    for row in s["confusion_auto_x_humain"]:
        print("  ", row)


def cmd_ab_make(a) -> None:
    pair = human.load_ab_pair(a.runs, a.data, human.parse_selection(a.a), human.parse_selection(a.b))
    info = human.make_ab(a.out, pair["pred_a"], pair["pred_b"], pair["ctx_last"], pair["name_a"], pair["name_b"],
                         n_pairs=min(a.pairs, len(pair["pred_a"])), n_reviewers=a.reviewers, seed=a.seed)
    print(f"{info['n_pairs']} paires entre {info['models'][0]} et {info['models'][1]} -> {a.out}/")
    print(f"Donner à chaque relecteur : pairs/, CONSIGNE.txt et SON fichier reviewer_k.csv. Garder key.csv.")


def cmd_ab_score(a) -> None:
    d = Path(a.dir)
    files = [d / r if not Path(r).exists() else Path(r) for r in a.responses]
    s = human.score_ab(d / "key.csv", files)
    _write_json(s, d / "ab_score.json")
    m0 = s["modeles"][0]
    pj = s[f"preference_{m0}_jugements"]
    pp = s[f"preference_{m0}_majorite_par_paire"]
    print(f"{s['reponses_valides']}/{s['reponses_attendues']} réponses valides ; modèles : {s['modeles']}")
    if pj:
        print(f"préférence pour {m0}, tous jugements : {pj['k']}/{pj['n']} = {pj['value']:.2f}  IC95 [{pj['ci_lo']:.2f} ; {pj['ci_hi']:.2f}]")
    if pp:
        print(f"préférence pour {m0}, majorité par paire : {pp['k']}/{pp['n']} = {pp['value']:.2f}  IC95 [{pp['ci_lo']:.2f} ; {pp['ci_hi']:.2f}]")
    print("part de réponses L par relecteur :", [round(x, 2) for x in s["part_de_reponses_L_par_relecteur"]], "(≈ 0,5 attendu)")
    print(f"κ moyen entre relecteurs : {s['kappa_moyen_entre_relecteurs']:.2f}")
    print(s["avertissement"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--runs", type=Path, default=Path("runs"))
        sp.add_argument("--data", type=Path, default=Path("data"))
        sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("review-make", description="100 cas stratifiés par classe automatique")
    common(sp)
    sp.add_argument("--select", nargs="+", required=True, help="run_id ou run_id:variante (pooler plusieurs modèles)")
    sp.add_argument("-n", type=int, default=100)
    sp.add_argument("--out", type=Path, default=Path("reports/human/review"))
    sp.set_defaults(fn=cmd_review_make)

    sp = sub.add_parser("review-score")
    sp.add_argument("--dir", type=Path, default=Path("reports/human/review"))
    sp.set_defaults(fn=cmd_review_score)

    sp = sub.add_parser("ab-make", description="paires A/B en aveugle")
    common(sp)
    sp.add_argument("--a", required=True)
    sp.add_argument("--b", required=True)
    sp.add_argument("--pairs", type=int, default=20)
    sp.add_argument("--reviewers", type=int, default=3)
    sp.add_argument("--out", type=Path, default=Path("reports/human/ab"))
    sp.set_defaults(fn=cmd_ab_make)

    sp = sub.add_parser("ab-score")
    sp.add_argument("--dir", type=Path, default=Path("reports/human/ab"))
    sp.add_argument("--responses", nargs="+", required=True, help="fichiers reviewer_k.csv remplis")
    sp.set_defaults(fn=cmd_ab_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
