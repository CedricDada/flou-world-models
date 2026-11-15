"""Description des entraînements : `RunSpec`, plan de 25 runs, estimation du temps. NumPy seulement.

Plan de 25 entraînements (cours, chap. 9) :

    cœur       9   3 familles x 3 graines, p = 0,3 (jetons : K = 256, grille 8 x 8)
    axe K      6   jetons, K in {64, 1024} x 3 graines, p = 0,3
    contrôles  3   voir ci-dessous
    axe p      6   3 familles x p in {0,1 ; 0,5} x 1 graine (exploratoire)
    grille     1   jetons, K = 256, grille 16 x 16, p = 0,3, 1 graine

Les trois contrôles : le cours en fixe le nombre (3) et les hypothèses H0c, H0d et « la perte à n = 1 n'est
pas propre à la stochasticité » (chap. 9, tableau « quelle notion pour quelle hypothèse »), pas la liste.
Lecture retenue ici :
    ctrl-oracle      régresseur oracle (reçoit la branche), p = 0,3 : H0c, moins de 5 % de flou
    ctrl-regression  régression à p = 0 : H0d, environ 0 % de flou
    ctrl-diffusion   diffusion à p = 0 : à n = 1 la perte de netteté ne doit pas venir de la diffusion elle-même
La grille de la ligne « grille : 1 » n'est pas définie non plus : on suppose une grille plus fine (16 x 16,
quatre fois plus de jetons) à K fixé, ce qui sépare l'effet de la grille de celui de K (cours, chap. 4).
Ce sont des lectures, modifiables ici sans toucher au reste.

Ordre d'exécution = ordre de priorité : cœur et contrôles d'abord (jamais sacrifiés), puis axe K, puis grille,
puis axe p (sacrifié en premier si le calcul manque).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..models.config import ModelSpec
from ..models.registry import default_spec

GROUPS = ("core", "control", "k_axis", "grid", "p_axis")
FAMILIES = ("regression", "oracle", "diffusion", "tokens")


@dataclass(frozen=True)
class RunSpec:
    family: str
    group: str
    p: float
    seed: int
    K: int | None = None            # jetons seulement
    grid: int = 8                   # jetons seulement
    k_ctx: int = 4
    steps: int = 20_000             # pas d'optimisation (jetons : VQ-VAE + Transformer)
    batch_size: int = 64
    lr: float = 2e-4
    vq_fraction: float = 0.4        # part des pas consacrée au VQ-VAE (famille C)
    ema: float = 0.0                # 0 : pas d'EMA (budget identique pour toutes les familles)
    note: str = ""

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"famille inconnue : {self.family}")
        if self.group not in GROUPS:
            raise ValueError(f"groupe inconnu : {self.group}")
        if (self.family == "tokens") != (self.K is not None):
            raise ValueError("K est requis (et seulement) pour la famille « tokens »")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p dans [0, 1]")

    @property
    def run_id(self) -> str:
        parts = [self.group.replace("_", ""), self.family]
        if self.K is not None:
            parts.append(f"K{self.K}")
            if self.grid != 8:
                parts.append(f"g{self.grid}")
        parts += [f"p{self.p:.2f}", f"s{self.seed}"]
        return "-".join(parts)

    def model_spec(self) -> ModelSpec:
        return default_spec(self.family, self.k_ctx, self.K or 256, self.grid)

    def stage_steps(self) -> dict[str, int]:
        if self.family == "tokens":
            vq = int(round(self.steps * self.vq_fraction))
            return {"vq": vq, "gpt": self.steps - vq}
        return {"main": self.steps}

    @property
    def images_seen(self) -> int:
        """Transitions vues à l'entraînement (même définition pour toutes les familles : budget apparié)."""
        return self.steps * self.batch_size

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, run_dir: str | Path) -> Path:
        path = Path(run_dir) / "spec.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, run_dir: str | Path) -> "RunSpec":
        return cls.from_dict(json.loads((Path(run_dir) / "spec.json").read_text(encoding="utf-8")))


def build_plan(steps: int = 20_000, batch_size: int = 64, lr: float = 2e-4, p_core: float = 0.3,
               p_axis: tuple[float, ...] = (0.1, 0.5), seeds: tuple[int, ...] = (0, 1, 2),
               grid_fine: int = 16, **common) -> list[RunSpec]:
    """Les 25 runs, dans l'ordre de priorité décroissante."""
    base = dict(steps=steps, batch_size=batch_size, lr=lr, **common)
    mk = lambda family, group, p, seed, **kw: RunSpec(family, group, p, seed, **base, **kw)
    plan: list[RunSpec] = []
    for seed in seeds:                                                 # cœur : 9
        plan += [mk("regression", "core", p_core, seed), mk("diffusion", "core", p_core, seed),
                 mk("tokens", "core", p_core, seed, K=256)]
    plan += [mk("oracle", "control", p_core, 0, note="H0c"),            # contrôles : 3
             mk("regression", "control", 0.0, 0, note="H0d"),
             mk("diffusion", "control", 0.0, 0, note="n = 1 sans bifurcation")]
    for seed in seeds:                                                 # axe K : 6
        plan += [mk("tokens", "k_axis", p_core, seed, K=64), mk("tokens", "k_axis", p_core, seed, K=1024)]
    plan.append(mk("tokens", "grid", p_core, 0, K=256, grid=grid_fine))  # grille : 1
    for p in p_axis:                                                   # axe p : 6
        plan += [mk("regression", "p_axis", p, 0), mk("diffusion", "p_axis", p, 0),
                 mk("tokens", "p_axis", p, 0, K=256)]
    ids = [r.run_id for r in plan]
    assert len(ids) == len(set(ids)), "identifiants de run en double"
    return plan


def summarize_plan(plan: list[RunSpec]) -> dict[str, int]:
    out = {g: 0 for g in GROUPS}
    for r in plan:
        out[r.group] += 1
    out["total"] = len(plan)
    return out


def estimate_hours(plan: list[RunSpec], seconds_per_step: dict[str, float]) -> float:
    """Heures de GPU prévues d'après des temps par pas mesurés (`scripts/smoke_test.py --time`)."""
    sec = 0.0
    for r in plan:
        for stage, n in r.stage_steps().items():
            key = r.family if r.family != "tokens" else f"tokens-{stage}"
            sec += n * seconds_per_step[key]
    return sec / 3600.0


def with_steps(plan: list[RunSpec], steps: int) -> list[RunSpec]:
    return [replace(r, steps=steps) for r in plan]


def data_dir_for(data_root: str | Path, p: float) -> Path:
    return Path(data_root) / f"p{p:.2f}"


def train_files(data_root: str | Path, p: float, split: str = "train") -> list[Path]:
    d = data_dir_for(data_root, p)
    files = sorted(d.glob(f"{split}_*.npz"))
    if not files:
        raise FileNotFoundError(f"aucun fichier {split}_*.npz dans {d} : lancer scripts/generate_data.py --split {split} --p {p}")
    return files


def preflight(plan: list[RunSpec], data_root: str | Path) -> dict[str, list[str]]:
    """Fichiers de données manquants pour exécuter le plan, AVANT de lancer un entraînement d'une heure et demie qui
    échouerait à l'évaluation. Renvoie {"manquants": [...], "avertissements": [...]} (listes vides si tout est là).
    Entraînement et test sont indispensables ; la validation est facultative (avertissement)."""
    missing, warn = [], []
    for p in sorted({r.p for r in plan}):
        d = data_dir_for(data_root, p)
        if not list(d.glob("train_*.npz")):
            missing.append(f"{d}/train_*.npz   (generate_data.py --split train --p {p})")
        if not list(d.glob("val_*.npz")):
            warn.append(f"{d}/val_*.npz   (generate_data.py --split val --p {p}) : pas de validation pour p = {p}")
        if not (d / "test_bifurcation.npz").exists():
            missing.append(f"{d}/test_bifurcation.npz   (generate_data.py --split test --p {p})")
    return {"manquants": missing, "avertissements": warn}


def test_file(data_root: str | Path, p: float) -> Path:
    path = data_dir_for(data_root, p) / "test_bifurcation.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} manquant : lancer scripts/generate_data.py --split test --p {p}")
    return path
