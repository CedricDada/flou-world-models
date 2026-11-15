import numpy as np
import pytest
from conftest import make_state

from flou.game import MiniGameEnv


def test_reset_and_step_follow_the_gymnasium_api(cfg):
    env = MiniGameEnv(cfg, seed=0)
    obs, info = env.reset(seed=1)
    assert obs.shape == (64, 64, 3) and obs.dtype == np.uint8
    assert info["state"].shape == (10,) and info["t"] == 0
    for t in range(cfg.max_steps):
        out = env.step(env.action_space.sample() if hasattr(env, "action_space") else t % 5)
        assert len(out) == 5
        obs, reward, terminated, truncated, info = out
        assert obs.shape == (64, 64, 3) and isinstance(reward, float)
        assert terminated is False and truncated == (t == cfg.max_steps - 1)
        assert set(info["events"]) >= {"blocked", "hit_enemy", "pickup", "bounce_x", "bounce_y"}
        assert info["branch"] in (-1, 0, 1)
    assert np.array_equal(env.render(), obs)


def test_reset_with_seed_is_reproducible(cfg):
    def play(seed):
        env = MiniGameEnv(cfg)
        env.reset(seed=seed)
        return np.stack([env.step(t % 5)[0] for t in range(30)])

    assert np.array_equal(play(3), play(3))
    assert not np.array_equal(play(3), play(4))


def test_reset_can_impose_the_initial_state(cfg):
    env = MiniGameEnv(cfg)
    s = make_state(cfg, ax=20, ay=20, eo=0, ev=-1)
    obs, info = env.reset(seed=0, options={"state": s.to_array()[0]})
    assert np.array_equal(info["state"], s.to_array()[0])
    _, _, _, _, info = env.step(0)
    assert info["branch"] in (0, 1)                        # on était au centre du rail


def test_gymnasium_spaces_when_installed(cfg):
    pytest.importorskip("gymnasium")
    env = MiniGameEnv(cfg)
    assert env.observation_space.shape == (64, 64, 3) and env.action_space.n == 5
    from gymnasium.utils.env_checker import check_env

    check_env(env.unwrapped if hasattr(env, "unwrapped") else env)
