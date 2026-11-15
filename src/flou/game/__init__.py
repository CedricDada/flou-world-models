"""Mini-jeu stochastique : 64x64 RGB, 5 actions, un ennemi qui bifurque avec probabilité p."""
from .config import (
    ACTION_NAMES, BRANCH_A, BRANCH_B, DOWN, LEFT, MIN_BRANCH_DIST, N_ACTIONS, NO_BRANCH,
    NOOP, RIGHT, UP, GameConfig,
)
from .dynamics import (
    advance, agent_step, box_gap, boxes_overlap, enemy_xy, geometry, initial_state,
    is_branch_state,
)
from .env import MiniGameEnv, VecMiniGame
from .render import branch_delta, render, render_counterfactual, render_one
from .state import STATE_DIM, Events, State

__all__ = [
    "ACTION_NAMES", "BRANCH_A", "BRANCH_B", "DOWN", "LEFT", "MIN_BRANCH_DIST", "N_ACTIONS",
    "NO_BRANCH", "NOOP", "RIGHT", "UP", "GameConfig", "advance", "agent_step", "box_gap",
    "boxes_overlap", "enemy_xy", "geometry", "initial_state", "is_branch_state",
    "MiniGameEnv", "VecMiniGame", "branch_delta", "render", "render_counterfactual",
    "render_one", "STATE_DIM", "Events", "State",
]
