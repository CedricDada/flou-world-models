import numpy as np
import pytest

from flou.data import (
    BifurcationSet, TransitionData, build_bifurcation_set, collect_episodes, count_duplicates,
    generate_split, stream_seeds,
)
from flou.game import GameConfig, State, advance, render


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    cfg = GameConfig(p=0.3)
    tr = generate_split(cfg, "train", ["random", "greedy"], 20_000, seed=0, out_dir=out, n_envs=64)
    va = generate_split(cfg, "val", ["random", "greedy"], 4_000, seed=0, out_dir=out, n_envs=64)
    return cfg, tr, va, out


def test_files_are_written_where_documented(files):
    cfg, tr, va, out = files
    assert [p.name for p in tr] == ["train_random.npz", "train_greedy.npz"]
    assert tr[0].parent == out / "p0.30"


def test_transition_count_and_bifurcation_share(files):
    cfg, tr, _, _ = files
    data = TransitionData(tr, k=4)
    assert len(data) == 20_000                           # 2 bots x 100 épisodes x 100 pas
    assert data.bifurcation_share() >= 0.05              # valeur proposée par le cours (chap. 9)
    assert abs(data.bifurcation_share() - 1 / 6) < 0.01


def test_stored_states_replay_exactly(files):
    """Rejouer (état, action, branche) avec `advance` redonne l'état suivant stocké : c'est ce
    qui garantit que les contrefactuels construits plus tard sont cohérents avec les données."""
    cfg, tr, _, _ = files
    data = TransitionData(tr, k=4)
    idx = np.arange(0, len(data), 7)
    b = data.get(idx, with_images=False)
    e, t = b["episode"], b["t"]
    s = State.from_array(data.states[e, t])
    nxt, ev = advance(s, data.actions[e, t], data.branch[e, t], cfg)
    assert np.array_equal(nxt.to_array(), data.states[e, t + 1])
    assert np.array_equal(ev.pickup, data.flags["pickup"][e, t])
    assert np.array_equal(ev.hit_enemy, data.flags["hit_enemy"][e, t])


def test_windows_and_images(files):
    cfg, tr, _, _ = files
    data = TransitionData(tr, k=4)
    b = data.get(np.array([0, 1, 2, 3, 4, 150]))
    assert b["context"].shape == (6, 4, 64, 64, 3) and b["target"].shape == (6, 64, 64, 3)
    # début d'épisode : le contexte répète s_0 (rembourrage)
    assert np.array_equal(b["context_states"][0], np.repeat(data.states[0, :1], 4, axis=0))
    assert np.array_equal(b["context_states"][3], data.states[0, 0:4])
    assert np.array_equal(b["context_states"][4], data.states[0, 1:5])
    # l'image cible est le rendu de l'état suivant
    assert np.array_equal(b["target"][3], render(State.from_array(data.states[0, 4]), cfg)[0])
    # la dernière image du contexte est celle de l'état courant
    assert np.array_equal(b["context"][5, -1], render(State.from_array(data.states[1, 50]), cfg)[0])


def test_generation_is_reproducible_and_splits_are_independent():
    cfg = GameConfig()
    a = collect_episodes(cfg, "greedy", 8, *stream_seeds(0, "train"), n_envs=8)
    b = collect_episodes(cfg, "greedy", 8, *stream_seeds(0, "train"), n_envs=8)
    c = collect_episodes(cfg, "greedy", 8, *stream_seeds(0, "val"), n_envs=8)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["states"], c["states"])
    assert len({stream_seeds(0, s, i) for s in ("train", "val", "test") for i in (0, 1)}) == 6


def test_train_val_do_not_share_episodes(files):
    cfg, tr, va, _ = files
    a, b = TransitionData(tr), TransitionData(va)
    assert count_duplicates(a.states[:, 0], b.states[:, 0]) == 0      # états initiaux distincts


def test_bifurcation_test_set(files, tmp_path):
    cfg, tr, _, _ = files
    ts = build_bifurcation_set(cfg, n=500, k=4, seed=0, n_envs=64)
    assert len(ts) == 500 and ts.ctx_states.shape == (500, 4, 10)
    assert (ts.ctx_states[:, -1, 6] == 0).all()                     # eo == 0 : états de bifurcation
    assert set(np.unique(ts.branch)) <= {0, 1}
    # reproductible et relisible
    again = build_bifurcation_set(cfg, n=500, k=4, seed=0, n_envs=64)
    assert np.array_equal(ts.ctx_states, again.ctx_states)
    path = ts.save(tmp_path / "t.npz", {"config": cfg.to_dict()})
    loaded = BifurcationSet.load(path)
    assert np.array_equal(loaded.ctx_states, ts.ctx_states) and loaded.cfg == cfg
    # contrefactuels : deux images distinctes, δ proche de 7,6
    img_a, img_b, delta = ts.counterfactuals()
    assert img_a.shape == (500, 64, 64, 3) and (img_a != img_b).any(axis=(1, 2, 3)).all()
    assert delta.min() > 6.0
    assert ts.contexts().shape == (500, 4, 64, 64, 3)
    # pas de fuite : aucun contexte du test ne figure dans l'entraînement
    train = TransitionData(tr, k=4)
    assert count_duplicates(train.context_states(np.arange(len(train))), ts.ctx_states) == 0


def test_empirical_branch_rate_in_the_test_set_matches_p():
    for p in (0.1, 0.5):
        cfg = GameConfig(p=p)
        ts = build_bifurcation_set(cfg, n=2000, k=4, seed=1, n_envs=128)
        assert abs(ts.branch.mean() - p) < 4 * np.sqrt(p * (1 - p) / 2000)


def test_mixed_config_files_are_rejected(tmp_path):
    a = generate_split(GameConfig(p=0.3), "val", ["random"], 200, 0, tmp_path / "a", n_envs=2)[0]
    b = generate_split(GameConfig(p=0.5), "val", ["random"], 200, 0, tmp_path / "b", n_envs=2)[0]
    with pytest.raises(ValueError, match="configuration"):
        TransitionData([a, b])
