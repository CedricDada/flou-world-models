#!/usr/bin/env python
"""Diagnostic exploratoire : OÙ se trouve l'écart des images classées « autre » (ρ > ρmax) ?

Prototype de la mesure proposée en J9, point 3. **Ce n'est pas une mesure préenregistrée** : la mesure
principale reste le classifieur sur l'image entière (`src/flou/eval/classifier.py`). Tant que
`docs/DECISIONS.md` n'a pas tranché, aucun verdict ne doit s'appuyer sur ce qui suit.

Pour chaque image conservée dans `eval_arrays.npz`, on calcule le reste r = ŷ − (α̂A + (1−α̂)B) et on
répartit son énergie ‖r‖² entre trois zones de l'image :

- « ennemi » : pixels où A et B diffèrent (le rail de la bifurcation), dilatés de 2 pixels ;
- « agent »  : pixels où l'agent est dessiné dans A (même position dans A et B), dilatés de 2 pixels ;
- « reste »  : tout le reste de l'image (fond, murs, petit objet).

Puis on recalcule α̂ et ρ en ne gardant que la zone « ennemi » (classifieur restreint au rail), ce qui
sépare deux questions : « la bifurcation est-elle tranchée, mélangée ou ratée ? » et « le reste de
l'image est-il juste ? ».

**Base de mesure.** `eval_arrays.npz` ne conserve que 25 images par classe. Les parts d'énergie et les
reclassements portent donc sur 25 cas au plus, pas sur les 2 000 états du jeu de test : de quoi établir
l'origine d'un écart, pas de quoi chiffrer une proportion.

Exemples
    uv run scripts/where_other.py
    uv run scripts/where_other.py --runs flou_etat/runs
    uv run scripts/where_other.py --runs flou_etat/runs core-diffusion-p0.30-s0:n=1
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np

CIBLES_DEFAUT = ["core-diffusion-p0.30-s0:n=1", "core-diffusion-p0.30-s0:n=3",
                 "core-tokens-K256-p0.30-s0:tau=1", "core-regression-p0.30-s0:main"]
NOMS = ["nette-A", "nette-B", "floue", "autre"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cibles", nargs="*", default=None, help="<run>:<variante>, par exemple core-diffusion-p0.30-s0:n=1")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--instrument", type=Path, default=Path("reports/instrument.json"),
                    help="d'où vient ρmax (surtout pas une valeur codée en dur : il dépend du jeu de test)")
    ap.add_argument("--p", type=float, default=0.3)
    args = ap.parse_args()

    from flou.data.dataset import BifurcationSet
    from flou.eval.classifier import alpha_rho, as_float, classify, dilate, rail_mask
    from flou.train import runs as R

    if not args.instrument.exists():
        raise SystemExit(f"{args.instrument} absent : lancer uv run scripts/validate_instrument.py")
    rho_max = json.loads(args.instrument.read_text(encoding="utf-8"))["rho_max"]

    bset = BifurcationSet.load(R.test_file(str(args.data), args.p))
    a_tout, b_tout, _ = bset.counterfactuals()
    a_tout, b_tout = as_float(a_tout), as_float(b_tout)
    print(f"jeu de test p = {args.p} : {len(a_tout)} états, images {a_tout.shape[1:]}, ρmax = {rho_max:.4f}")

    for cible in (args.cibles or CIBLES_DEFAUT):
        if ":" not in cible:
            raise SystemExit(f"cible mal formée : {cible!r} (attendu <run>:<variante>)")
        run, cle = cible.split(":", 1)
        chemin = args.runs / run / "eval_arrays.npz"
        if not chemin.exists():
            print(f"{cible} : {chemin} absent")
            continue
        z = np.load(chemin)
        if f"{cle}__img" not in z.files:
            variantes = sorted({k.split("__")[0] for k in z.files})
            print(f"{cible} : pas d'images conservées pour « {cle} » (variantes présentes : {', '.join(variantes)})")
            continue
        idx = z[f"{cle}__img_idx"]
        etat = z[f"{cle}__state"][idx]
        lab = z[f"{cle}__label"][idx]
        yhat = as_float(z[f"{cle}__img"])
        a, b = a_tout[etat], b_tout[etat]
        alpha, rho = alpha_rho(yhat, a, b)
        proj = alpha[:, None, None, None] * a + (1 - alpha[:, None, None, None]) * b
        res2 = ((yhat - proj) ** 2).sum(-1)                       # (N, H, W)
        diff = np.abs(a - b).sum(-1) > 1e-6                       # pixels de la bifurcation
        ennemi = rail_mask(a, b)
        # agent : pixels clairs identiques dans A et B (hors rail de l'ennemi)
        clair = (a.min(-1) > 0.7) & ~diff
        agent = dilate(clair) & ~ennemi
        reste = ~(ennemi | agent)
        tot = res2.sum((1, 2)) + 1e-12
        part = {nom: (res2 * m).sum((1, 2)) / tot
                for nom, m in (("ennemi", ennemi), ("agent", agent), ("reste", reste))}
        # classifieur restreint au rail de l'ennemi
        masque = ennemi[..., None]
        a_r, r_r = alpha_rho(yhat * masque, a * masque, b * masque)
        lab_r = classify(a_r, r_r, rho_max)
        print(f"\n== {cible}   ({len(idx)} images conservées, jusqu'à 25 par classe)")
        for c in range(4):
            sel = lab == c
            if not sel.any():
                continue
            deplace = np.bincount(lab_r[sel], minlength=4)
            print(f"  {NOMS[c]:8s} n={sel.sum():2d}  part de ‖reste‖² : ennemi {part['ennemi'][sel].mean():5.1%}"
                  f"  agent {part['agent'][sel].mean():5.1%}  reste {part['reste'][sel].mean():5.1%}"
                  f"   ρ médian {np.median(rho[sel]):.2f} | restreint au rail -> "
                  + ", ".join(f"{NOMS[k]} {deplace[k]}" for k in range(4) if deplace[k]))


if __name__ == "__main__":
    main()
