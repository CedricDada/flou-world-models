import numpy as np
import pytest

from flou.bots import BOTS, make_bot
from flou.game import N_ACTIONS, GameConfig, VecMiniGame


def run(bot_name, cfg, n_envs=128, episodes=2, seed=0):
    env = VecMiniGame(cfg, n_envs, seed=seed, render_obs=False)
    bot = make_bot(bot_name, cfg)
    rng = np.random.default_rng(seed + 1)
    pick = hits = bif = steps = 0
    for _ in range(episodes):
        env.reset()
        bot.reset(n_envs)
        for _ in range(cfg.max_steps):
            _, _, _, _, info = env.step(bot.act(env.state, rng))
            ev = info["events"]
            pick += ev.pickup.sum()
            hits += ev.hit_enemy.sum()
            bif += (ev.branch >= 0).sum()
            steps += n_envs
    n_ep = n_envs * episodes
    return dict(pickups=pick / n_ep, hits=hits / n_ep, bif=bif / steps)


@pytest.fixture(scope="module")
def stats():
    cfg = GameConfig()
    return {name: run(name, cfg) for name in BOTS}


def test_registry_has_the_four_bots():
    assert set(BOTS) == {"random", "greedy", "expert", "noisy_expert"}
    with pytest.raises(KeyError, match="inconnu"):
        make_bot("nope", GameConfig())


@pytest.mark.parametrize("name", sorted(BOTS))
def test_actions_are_valid_and_reproducible(name, cfg):
    env = VecMiniGame(cfg, 64, seed=0, render_obs=False)
    env.reset()
    bot = make_bot(name, cfg)
    a1 = bot.act(env.state, np.random.default_rng(5))
    a2 = bot.act(env.state, np.random.default_rng(5))
    assert a1.shape == (64,) and a1.dtype == np.int32
    assert a1.min() >= 0 and a1.max() < N_ACTIONS
    assert np.array_equal(a1, a2)


def test_greedy_collects_far_more_than_random(stats):
    assert stats["random"]["pickups"] < 0.5
    assert stats["greedy"]["pickups"] > 1.5
    assert stats["expert"]["pickups"] > 1.0


def test_expert_avoids_the_enemy(stats):
    assert stats["expert"]["hits"] < 0.2
    assert stats["greedy"]["hits"] > 10 * stats["expert"]["hits"]
    assert stats["greedy"]["hits"] > 2.0


def test_noisy_expert_sits_between_expert_and_greedy(stats):
    assert stats["expert"]["hits"] <= stats["noisy_expert"]["hits"] < stats["greedy"]["hits"]
    assert stats["noisy_expert"]["pickups"] > 1.0


def test_bifurcation_share_does_not_depend_on_the_bot(stats):
    # l'ennemi est autonome : une bifurcation tous les 2·spoke_len pas, soit 1/6 des transitions
    for name, s in stats.items():
        assert 0.16 < s["bif"] < 0.175, (name, s)
