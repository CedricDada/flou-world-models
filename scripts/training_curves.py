#!/usr/bin/env python
"""Courbes d'entraînement et alerte de surapprentissage. NumPy et matplotlib seulement, pas de PyTorch.

Lit les `runs/<run>/log.jsonl` écrits par le trainer et répond à une seule question : **ce run
a-t-il mémorisé son jeu d'entraînement ?** Deux symptômes sont cherchés.

1. **La validation remonte en tendance.** Signal universel, valable pour les quatre familles : si la
   pente de log(validation) sur la **seconde moitié** de l'entraînement est positive avec un t
   supérieur à `--seuil-t`, le modèle a commencé à mémoriser. C'est ce qui a été observé au premier
   run (incident J6 : pente +0,21 par 10 000 pas, t = +10,7 ; validation passée de 0,000803 au pas
   14 000 à 0,001195 à la fin).

   La règle a d'abord comparé le **dernier** point au minimum. Elle produisait des fausses alertes sur
   la diffusion, dont la validation fluctue fortement d'un relevé à l'autre même à bruit fixe, les
   poids changeant entre deux relevés : au run `core-diffusion-p0.30-s0`, +30 % sur le dernier point
   alors que la tendance de fond descendait encore (pente −0,40, t = −3,9). Voir `docs/JOURNAL.md`,
   J9. L'écart au minimum reste affiché, comme information.

2. **La perte d'entraînement passe sous le plancher de Bayes.** Vérification plus forte, mais qui
   n'a de sens que pour les familles `regression` et `oracle`, dont la perte est une MSE : aux
   états de bifurcation, l'erreur quadratique minimale d'un modèle qui ignore la branche vaut
   p(1-p)δ². Passer dessous est mathématiquement impossible sans connaître l'issue. Les pertes de
   la diffusion (débruitage) et des jetons (entropie croisée) n'ont pas de plancher comparable :
   pour elles, seul le symptôme 1 s'applique. Le plancher est estimé sur les données
   d'entraînement elles-mêmes, pas sur une valeur codée en dur.

Écrit une figure par run (`runs/<run>/fig_training.png`) et renvoie un **code de sortie non nul**
si au moins une alerte est levée, pour pouvoir enchaîner `&&` dans un script.

Exemples
    uv run scripts/training_curves.py --all
    uv run scripts/training_curves.py --run-id core-regression-p0.30-s0
    uv run scripts/training_curves.py --all --no-fig          # alertes seulement, sans figure
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np


def lire_log(path: Path) -> tuple[dict, dict]:
    """(train, val) : {stade: [(pas, perte)]} et {(stade, clé): [(pas, valeur)]}."""
    train: dict = {}
    val: dict = {}
    for ligne in path.read_text(encoding="utf-8").splitlines():
        if not ligne.strip():
            continue
        r = json.loads(ligne)
        stade, pas = r.get("stage", "main"), r.get("step")
        if pas is None:
            continue
        if "loss" in r:
            train.setdefault(stade, []).append((pas, r["loss"]))
        for k, v in r.items():
            if k.startswith("val_") and isinstance(v, (int, float)):
                val.setdefault((stade, k), []).append((pas, v))
    # une reprise réécrit quelques pas : on garde la dernière valeur de chaque pas, triée
    derniers = lambda pts: sorted({p: v for p, v in pts}.items())
    return {k: derniers(v) for k, v in train.items()}, {k: derniers(v) for k, v in val.items()}


def pente_log(pts: list[tuple[int, float]]) -> tuple[float | None, float]:
    """Pente de log(validation) sur la **seconde moitié** de l'entraînement, par 10 000 pas, et son t.

    Pourquoi la pente et non le dernier point : la validation de la diffusion fluctue fortement d'un
    relevé à l'autre, même à bruit fixe, parce que les poids changent entre deux relevés. Comparer le
    dernier point au minimum y produit des fausses alertes (incident J9 : +30 % sur un run dont la
    validation descendait encore). La tendance de fond, elle, sépare nettement les deux cas : pente
    de +0,21 par 10 000 pas pour le run mémorisant de J6, de −0,34 à −0,40 pour les diffusions saines.
    """
    pts = [(x, v) for x, v in pts if v > 0]
    if len(pts) < 6:
        return None, 0.0
    moitie = pts[len(pts) // 2:]
    x = np.array([p for p, _ in moitie], dtype=float) / 10_000.0
    y = np.log(np.array([v for _, v in moitie], dtype=float))
    if len(x) < 4 or np.ptp(x) == 0:
        return None, 0.0
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    s2 = float((resid ** 2).sum()) / (len(x) - 2)
    denom = float(((x - x.mean()) ** 2).sum())
    if s2 <= 0 or denom <= 0:
        return float(b), 0.0
    return float(b), float(b / np.sqrt(s2 / denom))


def plancher_bayes(data_root: Path, p: float, n_echantillon: int = 2000, seed: int = 0) -> float | None:
    """Plancher de Bayes de la MSE d'entraînement, dans les unités de `mse_loss` (moyenne par
    élément, échelle -1..1). None si les données d'entraînement sont absentes."""
    from flou.data.dataset import TransitionData
    from flou.game import State, branch_delta, render_counterfactual
    from flou.train.runs import train_files

    try:
        fichiers = train_files(data_root, p)
    except FileNotFoundError:                 # journaux analysés sans les données sous la main
        return None
    if not fichiers:
        return None
    td = TransitionData([str(f) for f in fichiers])
    part = td.bifurcation_share()
    bif = np.where(td.is_bifurcation())[0]
    if not len(bif):
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(bif, size=min(n_echantillon, len(bif)), replace=False)
    e, t = td.locate(idx)
    a, b, _, _ = render_counterfactual(State.from_array(td.states[e, t]), td.actions[e, t], td.cfg)
    delta2 = float((branch_delta(a, b) ** 2).mean())
    n_pixels = td.cfg.size * td.cfg.size * 3
    return part * p * (1.0 - p) * delta2 * 4.0 / n_pixels     # somme 0-1 -> moyenne par élément -1..1


def figure(run_id: str, train: dict, val: dict, plancher: float | None, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stades = list(train) or sorted({s for s, _ in val})
    fig, axes = plt.subplots(1, len(stades), figsize=(6 * len(stades), 4), squeeze=False)
    for ax, stade in zip(axes[0], stades):
        pts = train.get(stade, [])
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, lw=0.8, color="0.6", label="entraînement")
        for (s, cle), pv in sorted(val.items()):
            if s != stade:
                continue
            x, y = zip(*pv)
            ax.plot(x, y, lw=1.8, marker="o", ms=3, label=cle.replace("val_", "validation "))
        if plancher and stade == "main":
            ax.axhline(plancher, ls="--", lw=1.2, color="crimson", label=f"plancher de Bayes ({plancher:.2e})")
        ax.set_yscale("log")
        ax.set_xlabel("pas d'entraînement")
        ax.set_ylabel("perte")
        ax.set_title(f"{run_id} — stade « {stade} »")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def examiner(d: Path, data_root: Path, seuil_val: float, seuil_t: float, faire_fig: bool) -> list[str]:
    """Renvoie la liste des alertes (vide si tout va bien)."""
    from flou.train.runs import RunSpec

    log = d / "log.jsonl"
    if not log.exists():
        print(f"{d.name} : pas de log.jsonl, ignoré")
        return []
    spec = RunSpec.load(d)
    train, val = lire_log(log)
    alertes = []

    plancher = None
    if spec.family in ("regression", "oracle"):
        plancher = plancher_bayes(data_root, spec.p)
        pts = train.get("main", [])
        if plancher and len(pts) >= 10:
            fin = float(np.mean([v for _, v in pts[-10:]]))
            if fin < plancher:
                alertes.append(f"perte d'entraînement finale {fin:.6f} SOUS le plancher de Bayes "
                               f"{plancher:.6f} ({fin / plancher:.2f} x) : le modèle a mémorisé la branche")

    for (stade, cle), pv in sorted(val.items()):
        if len(pv) < 3:
            continue
        y = [v for _, v in pv]
        i_min = int(np.argmin(y))
        hausse = (y[-1] - y[i_min]) / y[i_min] if y[i_min] > 0 else 0.0
        b, t = pente_log(pv)
        monte = b is not None and b > 0 and t > seuil_t
        etat = "ALERTE" if monte else ("tendance ok" if hausse > seuil_val else "ok")
        pente_txt = "pente indisponible" if b is None else f"pente {b:+.3f}/10k pas, t {t:+.1f}"
        print(f"   {stade}/{cle} : minimum {y[i_min]:.6f} au pas {pv[i_min][0]}, "
              f"fin {y[-1]:.6f} ({hausse:+.1%})  |  {pente_txt}  [{etat}]")
        if monte:
            alertes.append(f"{cle} en hausse sur la seconde moitié de l'entraînement "
                           f"(pente {b:+.3f} par 10 000 pas, t = {t:+.1f}) : surapprentissage au stade « {stade} »")
        elif hausse > seuil_val:
            print(f"      information : la validation finit {hausse:+.1%} au-dessus de son minimum, mais la tendance "
                  f"de fond ne monte pas. Point isolé, pas une alerte (voir docs/JOURNAL.md, J9).")

    if faire_fig:
        out = d / "fig_training.png"
        figure(d.name, train, val, plancher, out)
        print(f"   figure : {out}")
    return alertes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id")
    g.add_argument("--all", action="store_true", help="tous les runs de --runs qui ont un log.jsonl")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--seuil-val", type=float, default=0.10,
                    help="hausse relative depuis le minimum au-delà de laquelle on le signale (information)")
    ap.add_argument("--seuil-t", type=float, default=3.0,
                    help="t minimal d'une pente de log(validation) positive pour déclencher l'alerte")
    ap.add_argument("--no-fig", action="store_true", help="n'écrit aucune figure")
    args = ap.parse_args()

    if args.run_id:
        dossiers = [args.runs / args.run_id]
    else:
        dossiers = sorted(d for d in args.runs.iterdir() if (d / "log.jsonl").exists()) if args.runs.is_dir() else []
    if not dossiers:
        raise SystemExit(f"aucun run avec un log.jsonl dans {args.runs}/")

    total = []
    for d in dossiers:
        print(f"== {d.name}")
        for a in examiner(d, args.data, args.seuil_val, args.seuil_t, not args.no_fig):
            total.append((d.name, a))

    print()
    if total:
        print(f"{len(total)} ALERTE(S) — ne pas engager la suite du plan avant d'avoir tranché :")
        for run, a in total:
            print(f"  {run} : {a}")
        print("Lire docs/JOURNAL.md (J6) : le premier run du projet a échoué exactement comme cela.")
        raise SystemExit(1)
    print(f"{len(dossiers)} run(s) examiné(s), aucun symptôme de surapprentissage")


if __name__ == "__main__":
    main()
