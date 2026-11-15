import numpy as np
import pytest
from conftest import make_state

from flou.bots import make_bot
from flou.game import (
    BRANCH_A, BRANCH_B, DOWN, LEFT, NO_BRANCH, NOOP, RIGHT, UP, GameConfig, VecMiniGame,
    advance, enemy_xy, geometry, initial_state, is_branch_state,
)


# ---------------------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------------------
def test_agent_moves_by_speed(cfg):
    s = make_state(cfg, ax=30, ay=10)
    for action, (dx, dy) in {NOOP: (0, 0), UP: (0, -2), DOWN: (0, 2), LEFT: (-2, 0), RIGHT: (2, 0)}.items():
        nxt, ev = advance(s, [action], [NO_BRANCH], cfg)
        assert (nxt.ax[0] - 30, nxt.ay[0] - 10) == (dx, dy)
        assert not ev.blocked[0]


def test_agent_blocked_by_outer_wall(cfg):
    s = make_state(cfg, ax=2, ay=10)                      # collé au mur de gauche
    nxt, ev = advance(s, [LEFT], [NO_BRANCH], cfg)
    assert nxt.ax[0] == 2 and ev.blocked[0]


def test_agent_blocked_by_pillar(cfg):
    s = make_state(cfg, ax=20, ay=14)                     # à droite du pilier (12..18, 12..18)
    nxt, ev = advance(s, [LEFT], [NO_BRANCH], cfg)        # 20 -> 18 : la boîte 18..22 touche à peine
    assert nxt.ax[0] == 18 and not ev.blocked[0]
    nxt2, ev2 = advance(nxt, [LEFT], [NO_BRANCH], cfg)    # 18 -> 16 : chevauche le pilier
    assert nxt2.ax[0] == 18 and ev2.blocked[0]


def test_agent_blocked_by_obstacle_only_when_entering_it(cfg):
    # obstacle 6x6 en (40, 40) ; l'agent (4x4) arrive par la gauche
    s = make_state(cfg, ax=34, ay=40, ox=40, oy=40, ovx=1, ovy=1)
    nxt, ev = advance(s, [RIGHT], [NO_BRANCH], cfg)
    assert nxt.ax[0] == 36 and not ev.blocked[0]          # boîte 36..40 : simple contact, pas de blocage
    nxt2, ev2 = advance(nxt, [RIGHT], [NO_BRANCH], cfg)
    assert nxt2.ax[0] == 36 and ev2.blocked[0]             # 38..42 chevaucherait l'obstacle : refusé
    # déjà chevauché (l'obstacle est venu sur l'agent) : l'agent doit pouvoir sortir
    s2 = make_state(cfg, ax=42, ay=42, ox=40, oy=40, ovx=1, ovy=1)
    out, ev3 = advance(s2, [RIGHT], [NO_BRANCH], cfg)
    assert out.ax[0] == 44 and not ev3.blocked[0]


# ---------------------------------------------------------------------------------------
# Obstacle
# ---------------------------------------------------------------------------------------
def test_obstacle_bounces_on_wall_and_flips_velocity(cfg):
    s = make_state(cfg, ox=3, oy=40, ovx=-2, ovy=1)       # mur de gauche en x = 0..1 : 3 - 2 = 1 touche
    nxt, ev = advance(s, [NOOP], [NO_BRANCH], cfg)
    assert ev.bounce_x[0] and not ev.bounce_y[0]
    assert nxt.ox[0] == 3 and nxt.ovx[0] == 2             # reste en place ce pas-là, vitesse inversée
    assert nxt.oy[0] == 41 and nxt.ovy[0] == 1            # l'autre axe est indépendant
    nxt2, ev2 = advance(nxt, [NOOP], [NO_BRANCH], cfg)
    assert nxt2.ox[0] == 5 and not ev2.bounce_x[0]


def test_obstacle_is_deterministic_and_never_stops(cfg):
    s = initial_state(cfg, 200, np.random.default_rng(0))
    g = geometry(cfg)
    for _ in range(300):                                  # plusieurs rebonds par partie
        s, _ev = advance(s, np.zeros(200, dtype=int), np.where(is_branch_state(s), 1, -1), cfg)
        assert not g.hits_wall(s.ox, s.oy, cfg.obstacle_size).any()
        assert (s.ovx != 0).all() and (s.ovy != 0).all()
        assert np.isin(np.abs(s.ovx), cfg.obstacle_speeds).all()


# ---------------------------------------------------------------------------------------
# Ennemi et bifurcation
# ---------------------------------------------------------------------------------------
def test_enemy_cycle_has_period_2L_and_branches_only_at_center(cfg):
    L = cfg.spoke_len
    s = make_state(cfg, eo=0, ev=-1)
    trace = []
    for branch in (BRANCH_A, BRANCH_B, BRANCH_A):
        for _ in range(2 * L):
            b = np.array([branch if is_branch_state(s)[0] else NO_BRANCH])
            s, ev = advance(s, [NOOP], b, cfg)
            trace.append(int(s.eo[0]))
    # A : 1 2 3 2 1 0 ; B : -1 -2 -3 -2 -1 0 ; A : 1 2 3 2 1 0
    assert trace == [1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0]


def test_branch_is_required_at_hub_and_ignored_elsewhere(cfg):
    hub = make_state(cfg, eo=0)
    with pytest.raises(ValueError, match="branche manquante"):
        advance(hub, [NOOP], [NO_BRANCH], cfg)
    away = make_state(cfg, eo=2, ev=1)
    a, _ = advance(away, [NOOP], [BRANCH_A], cfg)
    b, _ = advance(away, [NOOP], [BRANCH_B], cfg)
    assert a == b                                          # hors du centre la branche est ignorée


def test_branch_changes_only_the_enemy(cfg):
    """Aux états de bifurcation, A et B ne diffèrent que par l'ennemi (eo, ev)."""
    s = initial_state(cfg, 500, np.random.default_rng(1))
    s.eo[:] = 0
    act = np.random.default_rng(2).integers(0, 5, 500)
    sa, ea = advance(s, act, np.full(500, BRANCH_A), cfg)
    sb, eb = advance(s, act, np.full(500, BRANCH_B), cfg)
    for f in ("ax", "ay", "ox", "oy", "ovx", "ovy", "gx", "gy"):
        assert np.array_equal(getattr(sa, f), getattr(sb, f)), f
    assert (sa.eo == 1).all() and (sb.eo == -1).all()
    for f in ("blocked", "bounce_x", "bounce_y", "pickup", "hit_obstacle"):
        assert np.array_equal(getattr(ea, f), getattr(eb, f)), f
    assert (ea.branch == BRANCH_A).all() and (eb.branch == BRANCH_B).all()


def test_branches_are_at_least_min_distance_apart(cfg):
    sa, _ = advance(make_state(cfg, eo=0), [NOOP], [BRANCH_A], cfg)
    sb, _ = advance(make_state(cfg, eo=0), [NOOP], [BRANCH_B], cfg)
    (xa, ya), (xb, yb) = enemy_xy(sa.eo, cfg), enemy_xy(sb.eo, cfg)
    assert np.hypot(xa - xb, ya - yb)[0] >= 8


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.9])
def test_branch_frequency_matches_p_and_is_independent(p):
    cfg = GameConfig(p=p)
    env = VecMiniGame(cfg, 512, seed=7, render_obs=False)
    bot = make_bot("random", cfg)
    rng = np.random.default_rng(0)
    seq = []
    for _ in range(3):
        env.reset()
        run = []
        for _ in range(cfg.max_steps):
            _, _, _, _, info = env.step(bot.act(env.state, rng))
            run.append(info["branch"])
        seq.append(np.stack(run, axis=1))                  # (512, T)
    br = np.concatenate(seq)                               # (1536, T)
    drawn = br[br >= 0]
    n = len(drawn)
    assert abs(drawn.mean() - p) < 4 * np.sqrt(p * (1 - p) / n)
    # indépendance : fréquence de A sachant que la branche précédente (même partie) était A
    prev, nxt = [], []
    for row in br:
        r = row[row >= 0]
        prev.append(r[:-1]), nxt.append(r[1:])
    prev, nxt = np.concatenate(prev), np.concatenate(nxt)
    for value in (0, 1):
        sel = nxt[prev == value]
        assert abs(sel.mean() - p) < 4 * np.sqrt(p * (1 - p) / len(sel))


def test_p_extremes_are_deterministic():
    for p, expected in ((0.0, BRANCH_B), (1.0, BRANCH_A)):
        cfg = GameConfig(p=p)
        env = VecMiniGame(cfg, 64, seed=0, render_obs=False)
        env.reset()
        seen = set()
        for _ in range(cfg.max_steps):
            _, _, _, _, info = env.step(np.zeros(64, dtype=int))
            seen |= set(np.unique(info["branch"][info["branch"] >= 0]))
        assert seen == {expected}


# ---------------------------------------------------------------------------------------
# Petit objet
# ---------------------------------------------------------------------------------------
def test_pickup_respawns_at_farthest_spot_and_scores(cfg):
    s = make_state(cfg, ax=4, ay=4, gx=6, gy=6)            # l'agent chevauche le petit objet
    nxt, ev = advance(s, [NOOP], [NO_BRANCH], cfg)
    assert ev.pickup[0]
    assert (nxt.gx[0], nxt.gy[0]) == (56, 56)              # coin opposé
    ok, ev2 = advance(make_state(cfg, ax=20, ay=20, gx=6, gy=6), [NOOP], [NO_BRANCH], cfg)
    assert not ev2.pickup[0] and (ok.gx[0], ok.gy[0]) == (6, 6)


def test_gem_never_respawns_on_the_agent(cfg):
    rng = np.random.default_rng(3)
    s = initial_state(cfg, 4000, rng)
    s.ax, s.ay = rng.integers(2, 58, 4000).astype(np.int32), rng.integers(2, 58, 4000).astype(np.int32)
    s.gx, s.gy = s.ax.copy() + 1, s.ay.copy() + 1          # force un ramassage immédiat
    nxt, ev = advance(s, np.zeros(4000, dtype=int), np.where(is_branch_state(s), 1, -1), cfg)
    assert ev.pickup.all()
    d = np.hypot(nxt.gx - nxt.ax, nxt.gy - nxt.ay)
    assert (d >= 20).all()                                 # toujours loin de l'agent


# ---------------------------------------------------------------------------------------
# Environnement vectorisé et invariants
# ---------------------------------------------------------------------------------------
def test_same_seed_same_trajectory(cfg):
    def run(seed):
        env = VecMiniGame(cfg, 32, seed=seed, render_obs=False)
        env.reset()
        rng = np.random.default_rng(0)
        out = []
        for _ in range(50):
            _, _, _, _, info = env.step(rng.integers(0, 5, 32))
            out.append(info["state"].to_array())
        return np.stack(out)

    assert np.array_equal(run(5), run(5))
    assert not np.array_equal(run(5), run(6))


def test_batch_advance_equals_one_by_one(cfg):
    rng = np.random.default_rng(4)
    s = initial_state(cfg, 64, rng)
    act = rng.integers(0, 5, 64)
    br = np.where(is_branch_state(s), rng.integers(0, 2, 64), -1)
    batch, ev = advance(s, act, br, cfg)
    for i in range(64):
        one, evi = advance(s[i], act[i:i + 1], br[i:i + 1], cfg)
        assert one == batch[i]
        assert evi.pickup[0] == ev.pickup[i] and evi.blocked[0] == ev.blocked[i]


@pytest.mark.parametrize("bot_name", ["random", "greedy", "expert", "noisy_expert"])
def test_invariants_over_long_runs(cfg, bot_name):
    g = geometry(cfg)
    env = VecMiniGame(cfg, 128, seed=11, render_obs=False)
    bot = make_bot(bot_name, cfg)
    rng = np.random.default_rng(12)
    spots = {tuple(p) for p in cfg.gem_spots}
    L = cfg.spoke_len
    for _ in range(2):
        env.reset()
        for _ in range(cfg.max_steps):
            _, _, _, _, info = env.step(bot.act(env.state, rng))
            s = info["state"]
            assert not g.hits_wall(s.ax, s.ay, cfg.agent_size).any()
            assert not g.hits_wall(s.ox, s.oy, cfg.obstacle_size).any()
            assert (np.abs(s.eo) <= L).all() and np.isin(s.ev, (-1, 1)).all()
            assert ((np.abs(s.eo) != L) | (s.ev == -np.sign(s.eo))).all()
            assert {(int(x), int(y)) for x, y in zip(s.gx, s.gy)} <= spots


def test_episode_truncates_and_requires_reset(cfg):
    env = VecMiniGame(cfg, 4, seed=0, render_obs=False)
    env.reset()
    for t in range(cfg.max_steps):
        _, _, term, trunc, _ = env.step(np.zeros(4, dtype=int))
        assert not term.any() and trunc.all() == (t == cfg.max_steps - 1)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4, dtype=int))
    env.reset()
    assert env.t == 0 and (env.score == 0).all()


def test_reward_is_pickups_minus_hits(cfg):
    env = VecMiniGame(cfg, 1, seed=0, render_obs=False)
    env.reset(state=make_state(cfg, ax=4, ay=4, gx=6, gy=6, eo=2, ev=1))
    _, r, *_ = env.step(np.array([NOOP]))
    assert r[0] == 1.0
    # l'ennemi (eo=2, x=41..47, sens -1) arrive en x=35..41 et chevauche l'agent (x=34..38)
    env.reset(state=make_state(cfg, ax=34, ay=30, gx=6, gy=6, eo=2, ev=-1))
    _, r, *_ = env.step(np.array([NOOP]))
    assert r[0] == -1.0


def test_initial_state_has_no_overlap_and_valid_enemy(cfg):
    s = initial_state(cfg, 3000, np.random.default_rng(9))
    from flou.game import boxes_overlap

    ex, ey = enemy_xy(s.eo, cfg)
    A = cfg.agent_size
    assert not boxes_overlap(s.ax, s.ay, A, s.ox, s.oy, cfg.obstacle_size).any()
    assert not boxes_overlap(s.ax, s.ay, A, ex, ey, cfg.enemy_size).any()
    assert not boxes_overlap(s.ax, s.ay, A, s.gx, s.gy, cfg.gem_size).any()
    assert ((np.abs(s.eo) != cfg.spoke_len) | (s.ev == -np.sign(s.eo))).all()
