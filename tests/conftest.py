import numpy as np
import pytest

from flou.game import GameConfig, State


@pytest.fixture
def cfg() -> GameConfig:
    return GameConfig()


def make_state(cfg: GameConfig, **over) -> State:
    """État valide à un seul élément, modifiable champ par champ (tests ciblés)."""
    base = dict(ax=32, ay=10, ox=40, oy=40, ovx=1, ovy=1, eo=2, ev=1, gx=6, gy=6)
    base.update(over)
    return State(**{k: np.array([v]) for k, v in base.items()})
