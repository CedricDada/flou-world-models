"""Paramètres du mini-jeu.

Toutes les valeurs sont regroupées ici. Celles que le cours préparatoire (chap. 9) donne comme
« proposées » (distance minimale entre branches, part de bifurcations, etc.) se modifient
sans toucher au reste du code : la configuration est sérialisée avec chaque jeu de
données, ce qui rend les expériences relisibles.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# ---- actions (5 actions discrètes) ---------------------------------------------------
NOOP, UP, DOWN, LEFT, RIGHT = range(5)
ACTION_NAMES = ("noop", "haut", "bas", "gauche", "droite")
N_ACTIONS = 5

# ---- branches de l'ennemi --------------------------------------------------------------
NO_BRANCH = -1  # pas d'état de bifurcation à ce pas
BRANCH_B = 0    # l'ennemi part à gauche  (probabilité 1 - p)
BRANCH_A = 1    # l'ennemi part à droite  (probabilité p)

# Distance minimale (pixels) entre les deux positions possibles de l'ennemi : valeur
# proposée par le cours préparatoire (chap. 9) pour que l'écart entre les images A et B
# domine le bruit de rendu.
MIN_BRANCH_DIST = 8

_TUPLE_FIELDS = (
    "pillars", "obstacle_speeds", "gem_spots",
    "color_bg", "color_wall", "color_obstacle", "color_enemy", "color_agent", "color_gem",
)


def _rect_overlap(a: tuple, b: tuple) -> bool:
    """Chevauchement de deux rectangles (x, y, largeur, hauteur)."""
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


@dataclass(frozen=True)
class GameConfig:
    # --- monde (pixels) ---
    size: int = 64                    # image 64 x 64
    wall: int = 2                     # épaisseur du mur d'enceinte
    pillars: tuple = ((12, 12, 6, 6), (46, 12, 6, 6), (12, 46, 6, 6), (46, 46, 6, 6))  # (x, y, l, h)
    # --- agent ---
    agent_size: int = 4
    agent_speed: int = 2              # pixels par pas
    # --- obstacle mobile déterministe ---
    obstacle_size: int = 6
    obstacle_speeds: tuple = (1, 2)   # |vx| et |vy| tirés dans cet ensemble, signes aléatoires
    # --- ennemi à bifurcation stochastique ---
    enemy_size: int = 6
    enemy_step: int = 6               # pixels par pas le long du rail
    spoke_len: int = 3                # longueur d'un bras du rail, en pas
    p: float = 0.3                    # probabilité de la branche A (droite)
    # --- petit objet à ramasser ---
    gem_size: int = 2
    gem_spots: tuple = ((6, 6), (56, 6), (6, 56), (56, 56), (31, 6), (31, 56), (6, 31), (56, 31))
    # --- épisode ---
    max_steps: int = 100
    # --- couleurs (RGB) ---
    color_bg: tuple = (12, 12, 20)
    color_wall: tuple = (80, 80, 95)
    color_obstacle: tuple = (200, 200, 210)
    color_enemy: tuple = (235, 55, 55)
    color_agent: tuple = (50, 130, 255)
    color_gem: tuple = (255, 215, 40)

    # ------------------------------------------------------------------------------------
    @property
    def hub(self) -> tuple[int, int]:
        """Coin haut-gauche de l'ennemi quand il est au centre du rail (état de bifurcation)."""
        h = (self.size - self.enemy_size) // 2
        return h, h

    @property
    def branch_distance(self) -> int:
        """Distance (pixels) entre les deux positions possibles de l'ennemi après bifurcation."""
        return 2 * self.enemy_step

    def with_p(self, p: float) -> "GameConfig":
        return GameConfig.from_dict({**self.to_dict(), "p": float(p)})

    # ------------------------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p doit être dans [0, 1]")
        if self.branch_distance < MIN_BRANCH_DIST:
            raise ValueError(
                f"distance entre branches {self.branch_distance} px < {MIN_BRANCH_DIST} px "
                "(augmenter enemy_step)"
            )
        if self.spoke_len < 1:
            raise ValueError("spoke_len doit être >= 1")
        hx, hy = self.hub
        rail = (hx - self.spoke_len * self.enemy_step, hy,
                2 * self.spoke_len * self.enemy_step + self.enemy_size, self.enemy_size)
        inner = (self.wall, self.wall, self.size - 2 * self.wall, self.size - 2 * self.wall)
        if not (inner[0] <= rail[0] and rail[0] + rail[2] <= inner[0] + inner[2]):
            raise ValueError("le rail de l'ennemi sort de l'arène")
        for pil in self.pillars:
            if _rect_overlap(rail, pil):
                raise ValueError(f"le pilier {pil} coupe le rail de l'ennemi")
        for sx, sy in self.gem_spots:
            box = (sx, sy, self.gem_size, self.gem_size)
            if not (inner[0] <= sx and sx + self.gem_size <= inner[0] + inner[2]
                    and inner[1] <= sy and sy + self.gem_size <= inner[1] + inner[3]):
                raise ValueError(f"emplacement du petit objet {box} hors de l'arène")
            if any(_rect_overlap(box, pil) for pil in self.pillars):
                raise ValueError(f"emplacement du petit objet {box} dans un pilier")
            if _rect_overlap(box, rail):
                raise ValueError(f"emplacement du petit objet {box} sur le rail de l'ennemi")

    # ------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, d: dict) -> "GameConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"paramètres inconnus : {sorted(unknown)}")
        out = {}
        for k, v in d.items():
            if k in _TUPLE_FIELDS:
                v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
            out[k] = v
        return cls(**out)

    @classmethod
    def from_json(cls, path_or_text: str | Path) -> "GameConfig":
        p = Path(str(path_or_text))
        text = p.read_text(encoding="utf-8") if p.exists() else str(path_or_text)
        return cls.from_dict(json.loads(text))
