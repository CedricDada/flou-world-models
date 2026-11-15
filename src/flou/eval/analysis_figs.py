"""Figures de l'analyse (matplotlib, lecture seule des enregistrements de `analysis.py`).

Règles : une couleur FIXE par famille (les trois premiers créneaux de la palette catégorielle validée pour
le daltonisme, avec en plus une forme de marqueur par famille) ; points bruts de chaque graine toujours
montrés, moyenne en trait fin ; étiquettes directes (le turquoise est sous 3:1 de contraste sur fond clair,
d'où des étiquettes visibles et les tables CSV jointes) ; grille discrète, pas d'axe double.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import analysis as A
from ..models import edm_math as E

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"regression": "#2a78d6", "diffusion": "#eb6834", "tokens": "#1baf7a", "oracle": "#6b6a66"}
MARK = {"regression": "o", "diffusion": "s", "tokens": "^", "oracle": "D"}
LABEL = {"regression": "A. Régression", "diffusion": "B. Diffusion", "tokens": "C. Jetons", "oracle": "Oracle"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "figure.dpi": 130, "savefig.bbox": "tight"})
    return plt


def _points(ax, x, ys, family, jitter=0.07, seed=0, **kw):
    rng = np.random.default_rng(seed)
    xs = x + rng.uniform(-jitter, jitter, len(ys))
    ax.scatter(xs, ys, s=22, marker=MARK[family], color=COLOR[family], edgecolor="#fcfcfb", linewidth=0.8,
               zorder=3, **kw)
    if len(ys):
        ax.hlines(np.mean(ys), x - 0.18, x + 0.18, color=COLOR[family], linewidth=1.4, zorder=2)


def fig_blur(recs, out: Path, p=0.3) -> Path | None:
    """Part d'images floues par condition à p donné (H1, H2a) : un point par graine, seuils de H1 en pointillés."""
    cond = []
    for fam, kw in (("regression", {}), ("diffusion", {"n_steps": 1}), ("diffusion", {"n_steps": 3}),
                    ("diffusion", {"n_steps": 5}), ("diffusion", {"n_steps": 20}), ("tokens", {"K": 256})):
        rr = A.select(recs, group="core", family=fam, p=p, **kw)
        if rr:
            name = LABEL[fam] + (f"\nn={kw['n_steps']}" if "n_steps" in kw else f"\nK={kw['K']}" if "K" in kw else "")
            cond.append((name, fam, [r["blur"] for r in rr]))
    if not cond:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, (name, fam, ys) in enumerate(cond):
        _points(ax, i, ys, fam)
        ax.annotate(f"{100 * np.mean(ys):.0f} %", (i, np.mean(ys)), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1)
    ax.text(len(cond) - 0.45, 0.51, "seuil H1 : régression > 50 %", ha="right", fontsize=7.5, color=MUTED)
    ax.axhline(0.10, color=MUTED, linestyle=":", linewidth=1)
    ax.text(len(cond) - 0.45, 0.11, "H1 réfutée si diffusion n=3 > 10 %", ha="right", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(cond)), [c[0] for c in cond])
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel("part d'images floues")
    ax.set_title(f"Flou par condition, p = {p:g} (un point par graine)", loc="left", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_blur_vs_p(recs, out: Path) -> Path | None:
    """Axe p (exploratoire, 1 graine par point, sauf p = 0,3 : moyenne des graines du cœur)."""
    series = {}
    for fam, kw in (("regression", {}), ("diffusion", {"n_steps": 3}), ("tokens", {"K": 256})):
        pts = {}
        for p in (0.1, 0.3, 0.5):
            grp = "core" if abs(p - 0.3) < 1e-9 else "p_axis"
            rr = A.select(recs, group=grp, family=fam, p=p, **kw)
            if rr:
                pts[p] = float(np.mean([r["blur"] for r in rr]))
        if pts:
            series[fam] = pts
    if not series:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ps = np.linspace(0, 1, 201)
    ax.fill_between(ps, 0, 1, where=(ps > 0.2) & (ps < 0.8), color=GRID, alpha=0.6, linewidth=0)
    ax.text(0.5, 0.03, "régresseur parfait classé « flou » (0,2 < p < 0,8)", ha="center", fontsize=7.5, color=MUTED)
    for fam, pts in series.items():
        xs = sorted(pts)
        ax.plot(xs, [pts[x] for x in xs], color=COLOR[fam], linewidth=1.4, marker=MARK[fam], markersize=5)
        ax.annotate(LABEL[fam] + (" (n=3)" if fam == "diffusion" else ""), (xs[-1], pts[xs[-1]]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, va="center", color=INK)
    ax.set_xlim(0.05, 0.75)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("probabilité de la branche A, p")
    ax.set_ylabel("part d'images floues")
    ax.set_title("Flou selon p (exploratoire)", loc="left", fontsize=10)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_calibration(recs, out: Path, p=0.3, delta=15.0) -> Path | None:
    """p̂ en fonction de n, avec la référence du débruiteur EXACT (cas jouet, σmax/δ ≈ 80/15 ≈ 5) : l'écart restant est
    l'erreur d'apprentissage, l'écart de la référence à p est le biais de l'échantillonneur d'Euler seul."""
    rr = A.select(recs, group="core", family="diffusion", p=p)
    rr = [r for r in rr if r["n_steps"] and not np.isnan(r["p_hat"])]
    if not rr:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ns_ref = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50]
    ref = [A_toy(p, n, 80.0 / delta) for n in ns_ref]
    ax.plot(ns_ref, ref, color=MUTED, linestyle="--", linewidth=1.2)
    ax.annotate("débruiteur exact\n(cas jouet, σmax = 5,3 δ)", (ns_ref[-1], ref[-1]), xytext=(-4, -30),
                textcoords="offset points", ha="right", fontsize=7.5, color=MUTED)
    ax.axhspan(p - 0.05, p + 0.05, color=GRID, alpha=0.7, linewidth=0)
    ax.axhline(p, color=MUTED, linewidth=0.8)
    ax.text(ns_ref[-1], p + 0.055, f"p = {p:g}  (bande ± 0,05 : seuil de H2b)", ha="right", fontsize=7.5, color=MUTED)
    for n in sorted({r["n_steps"] for r in rr}):
        ys = [r["p_hat"] for r in rr if r["n_steps"] == n]
        _points(ax, n, ys, "diffusion", jitter=0.02 * n)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 20, 50], ["1", "3", "5", "20", "50"])
    ax.set_xlim(0.8, 60)
    ax.set_ylim(-0.02, 0.42)
    ax.set_xlabel("pas d'Euler n (même débruiteur par graine)")
    ax.set_ylabel("p̂ sur les images nettes")
    ax.set_title("Calibration de la bifurcation selon n", loc="left", fontsize=10)
    fig.savefig(out)
    plt.close(fig)
    return out


def A_toy(p: float, n: int, sigma_max: float) -> float:
    return E.toy_p_hat(E.euler_sample_toy(p, n, 100_000, sigma_max=sigma_max))[0]


def fig_recall(recs, out: Path, p=0.3) -> Path | None:
    """Rappel du petit objet selon K (H3a), points = graines ; trait pointillé : rappel après simple reconstruction
    par le VQ-VAE (capacité du codeur-décodeur, sans Transformer)."""
    rows = []
    for K in (64, 256, 1024):
        rr = [r for r in A.select(recs, family="tokens", K=K, p=p, grid=8) if r["group"] in ("core", "k_axis")]
        if rr:
            rows.append((K, [r["recall"] for r in rr], [r["vq_recall"] for r in rr]))
    if not rows:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for i, (K, ys, vq) in enumerate(rows):
        _points(ax, i, ys, "tokens")
        vq = [v for v in vq if not np.isnan(v)]
        if vq:
            ax.scatter([i] * len(vq), vq, s=16, marker="_", color=MUTED, zorder=2)
    ax.plot(range(len(rows)), [np.mean(r[1]) for r in rows], color=COLOR["tokens"], linewidth=1.2)
    ax.set_xticks(range(len(rows)), [f"K = {r[0]}" for r in rows])
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("rappel du petit objet 2×2")
    ax.text(len(rows) - 0.55, 0.04, "tirets gris : reconstruction par le VQ-VAE seul", ha="right", fontsize=7.5, color=MUTED)
    ax.set_title(f"Rappel du petit objet selon K, p = {p:g}", loc="left", fontsize=10)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_rollout(recs, out: Path, p=0.3) -> Path | None:
    """Précision de suivi de l'agent et de l'obstacle par pas de rollout (accumulation d'erreur, taxonomie (d))."""
    curves = {}
    for fam, kw in (("regression", {}), ("diffusion", {"n_steps": 3}), ("tokens", {"K": 256})):
        rr = [r for r in A.select(recs, group="core", family=fam, p=p, **kw) if r["track_curve"]]
        if rr:
            curves[fam] = np.array([r["track_curve"] for r in rr])
    if not curves:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for fam, c in curves.items():
        h = np.arange(1, c.shape[1] + 1)
        for row in c:
            ax.plot(h, row, color=COLOR[fam], linewidth=0.7, alpha=0.45)
        ax.plot(h, c.mean(0), color=COLOR[fam], linewidth=1.6, marker=MARK[fam], markersize=3.5)
        ax.annotate(LABEL[fam], (h[-1], c.mean(0)[-1]), xytext=(5, 0), textcoords="offset points", fontsize=8,
                    va="center")
    ax.set_xlim(1, 19)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("pas de rollout")
    ax.set_ylabel("agent et obstacle à moins de 3 px")
    ax.set_title("Accumulation d'erreur sur 16 pas (traits fins : graines)", loc="left", fontsize=10)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_mse_floor(recs, out: Path, p=0.3) -> Path | None:
    """MSE / plancher de Bayes en espérance (H0b) : bande [1,00 ; 1,10] pour la régression."""
    items = []
    for fam, kw in (("regression", {}), ("diffusion", {"n_steps": 3}), ("tokens", {"K": 256})):
        rr = A.select(recs, group="core", family=fam, p=p, **kw)
        if rr:
            items.append((fam, [r["ratio_expected"] for r in rr]))
    if not items:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.axhspan(1.0, 1.1, color=GRID, alpha=0.9, linewidth=0)
    ax.text(len(items) - 0.5, 1.12, "bande H0b : 1,00 à 1,10", ha="right", fontsize=7.5, color=MUTED)
    ax.axhline(2.0, color=MUTED, linestyle=":", linewidth=1)
    ax.text(len(items) - 0.5, 2.03, "échantillonneur parfait ≈ 2", ha="right", fontsize=7.5, color=MUTED)
    for i, (fam, ys) in enumerate(items):
        _points(ax, i, ys, fam)
    ax.set_xticks(range(len(items)), [LABEL[f] for f, _ in items])
    ax.set_ylabel("MSE / plancher de Bayes")
    ax.set_title(f"Erreur quadratique contre le plancher, p = {p:g}", loc="left", fontsize=10)
    fig.savefig(out)
    plt.close(fig)
    return out


def all_figures(recs, out_dir: str | Path, p: float = 0.3) -> list[Path]:
    out_dir = Path(out_dir)
    made = [fig_blur(recs, out_dir / "fig_flou.png", p), fig_blur_vs_p(recs, out_dir / "fig_flou_selon_p.png"),
            fig_calibration(recs, out_dir / "fig_calibration.png", p), fig_recall(recs, out_dir / "fig_rappel.png", p),
            fig_rollout(recs, out_dir / "fig_rollout.png", p), fig_mse_floor(recs, out_dir / "fig_mse_plancher.png", p)]
    return [m for m in made if m is not None]
