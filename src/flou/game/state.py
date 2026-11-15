"""État exact du jeu (structure de tableaux, un élément par partie en parallèle).

Les images ne contiennent PAS la branche tirée à l'instant t : elle n'apparaît qu'en t+1.
C'est pourquoi l'état ne contient aucun champ « branche » : la branche est un argument de
`advance`, pas une partie de l'état.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np

_I32 = np.int32


@dataclass
class State:
    """Champs (tous des entiers, forme (B,)) :

    ax, ay    agent : coin haut-gauche (pixels, y vers le bas)
    ox, oy    obstacle mobile : coin haut-gauche
    ovx, ovy  obstacle : vitesse (pixels par pas), jamais nulle
    eo        ennemi : décalage sur le rail, en pas, de -spoke_len à +spoke_len (0 = centre)
    ev        ennemi : sens de déplacement, +1 (vers la droite) ou -1 (vers la gauche)
    gx, gy    petit objet : coin haut-gauche
    """

    ax: np.ndarray
    ay: np.ndarray
    ox: np.ndarray
    oy: np.ndarray
    ovx: np.ndarray
    ovy: np.ndarray
    eo: np.ndarray
    ev: np.ndarray
    gx: np.ndarray
    gy: np.ndarray

    FIELDS: ClassVar[tuple[str, ...]] = (
        "ax", "ay", "ox", "oy", "ovx", "ovy", "eo", "ev", "gx", "gy",
    )

    def __post_init__(self) -> None:
        for name in self.FIELDS:
            setattr(self, name, np.atleast_1d(np.asarray(getattr(self, name), dtype=_I32)))

    # -------------------------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.ax.shape[0])

    def copy(self) -> "State":
        return State(*(getattr(self, f).copy() for f in self.FIELDS))

    def __getitem__(self, idx) -> "State":
        """Sous-ensemble de parties (entier, tranche, masque ou liste d'indices)."""
        return State(*(np.atleast_1d(getattr(self, f)[idx]) for f in self.FIELDS))

    def to_array(self, dtype=np.int16) -> np.ndarray:
        """Matrice (B, 10) ; l'ordre des colonnes est State.FIELDS."""
        return np.stack([getattr(self, f) for f in self.FIELDS], axis=1).astype(dtype)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "State":
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None]
        if arr.shape[-1] != len(cls.FIELDS):
            raise ValueError(f"attendu {len(cls.FIELDS)} colonnes, reçu {arr.shape}")
        return cls(*(arr[..., i].astype(_I32) for i in range(len(cls.FIELDS))))

    @classmethod
    def concatenate(cls, states: list["State"]) -> "State":
        return cls(*(np.concatenate([getattr(s, f) for s in states]) for f in cls.FIELDS))

    def __eq__(self, other: object) -> bool:  # égalité exacte, pratique dans les tests
        if not isinstance(other, State):
            return NotImplemented
        return all(np.array_equal(getattr(self, f), getattr(other, f)) for f in self.FIELDS)

    __hash__ = None  # type: ignore[assignment]


STATE_DIM = len(State.FIELDS)


@dataclass
class Events:
    """Drapeaux d'événements d'un pas (tous de forme (B,))."""

    blocked: np.ndarray       # l'agent voulait bouger et a été arrêté (mur, pilier ou obstacle)
    hit_enemy: np.ndarray     # l'agent chevauche l'ennemi après le pas
    hit_obstacle: np.ndarray  # l'agent chevauche l'obstacle après le pas
    bounce_x: np.ndarray      # l'obstacle a rebondi sur un mur vertical
    bounce_y: np.ndarray      # l'obstacle a rebondi sur un mur horizontal
    pickup: np.ndarray        # le petit objet a été ramassé (il réapparaît ailleurs)
    branch: np.ndarray        # int8 : -1 aucune, 0 = B (gauche), 1 = A (droite)

    NAMES: ClassVar[tuple[str, ...]] = (
        "blocked", "hit_enemy", "hit_obstacle", "bounce_x", "bounce_y", "pickup", "branch",
    )

    def to_dict(self) -> dict[str, np.ndarray]:
        return {n: getattr(self, n) for n in self.NAMES}

    def __getitem__(self, idx) -> "Events":
        return Events(*(np.atleast_1d(getattr(self, n)[idx]) for n in self.NAMES))


def _check_fields() -> None:  # garde-fou : l'ordre de FIELDS suit celui du dataclass
    declared = tuple(f.name for f in fields(State))
    assert declared == State.FIELDS, (declared, State.FIELDS)


_check_fields()
