"""Extension « décision », niveau 1 : lecteur de contact, coût exact, erreur d'estimation (H0e, H4).

Tout est testé sur le vrai jeu et sur des prédicteurs factices dont la bonne réponse est connue ; aucun modèle appris.
"""
import numpy as np
import pytest

from conftest import make_state
from flou.data.dataset import build_bifurcation_set
from flou.eval import decision as D
from flou.eval import decision_instrument as DI
from flou.game import GameConfig, render
from flou.game.dynamics import advance, enemy_xy, is_branch_state
from flou.game.state import State


@pytest.fixture(scope="module")
def bset():
    return build_bifurcation_set(GameConfig().with_p(0.3), n=600, seed=1000)


@pytest.fixture(scope="module")
def sub(bset):
    return bset.subset(D.select_states(bset, (1, 3, 5), seed=0))


# ----------------------------------------------------------------------------- le jeu justifie « exact pour k <= 5 »
def test_no_second_bifurcation_within_five_steps(bset):
    cfg = bset.cfg
    assert D.max_exact_horizon(cfg) == 5
    for branch in (0, 1):
        cur = bset.last_state()
        for i in range(6):
            br = np.where(is_branch_state(cur), branch, -1)
            cur, _ = advance(cur, bset.future_actions[:, i], br, cfg)
            assert is_branch_state(cur).all() == (i == 5)           # retour au centre du rail au 6e pas seulement
            assert is_branch_state(cur).any() == (i == 5)


def test_branch_paths_first_step_is_the_counterfactual_pair(bset):
    paths = D.branch_paths(bset, 3)
    s_a, s_b = bset.next_states()
    assert np.array_equal(paths["states_a"][:, 0], s_a.to_array())
    assert np.array_equal(paths["states_b"][:, 0], s_b.to_array())
    # seul l'ennemi dépend de la branche, à tous les pas (colonnes eo, ev)
    other = [i for i, f in enumerate(State.FIELDS) if f not in ("eo", "ev")]
    assert np.array_equal(paths["states_a"][:, :, other], paths["states_b"][:, :, other])


def test_true_costs_are_the_expectation_over_the_branch(bset):
    t = D.true_costs(bset, 5)
    p = bset.cfg.p
    assert np.allclose(t["cost"], p * t["cost_a"] + (1 - p) * t["cost_b"])
    assert np.array_equal(t["ambiguous"], t["cost_a"] != t["cost_b"])
    assert (np.diff(t["cost_a"], axis=1) >= 0).all()                # cumulé : croissant avec l'horizon
    share = t["ambiguous"][:, 0].mean()
    assert 0.04 < share < 0.15                                      # ordre de grandeur mesuré : ~9 % à k = 1


def test_horizon_guard(bset):
    with pytest.raises(ValueError, match="seconde bifurcation"):
        D.branch_paths(bset, 6)
    with pytest.raises(ValueError):
        D.branch_paths(bset, 0)


def test_select_states_keeps_every_ambiguous_state(bset):
    idx = D.select_states(bset, (1, 3, 5), seed=0)
    amb = D.true_costs(bset, 5)["ambiguous"][:, [0, 2, 4]].any(axis=1)
    assert set(np.nonzero(amb)[0]) <= set(idx)
    assert len(idx) == 2 * amb.sum() and len(set(idx)) == len(idx)
    assert len(D.select_states(bset, (1,), n_other=0)) == D.true_costs(bset, 1)["ambiguous"][:, 0].sum()


# ----------------------------------------------------------------------------- lecteur de contact
def test_h0e_reader_recovers_the_game_event_on_true_frames(bset):
    rep = D.check_reader(bset)
    assert rep["pass"] and rep["agreement"] == 1.0 and rep["agent_found"] == 1.0
    assert rep["n_contacts"] > 50 and rep["soft_max_abs_err"] < 0.02


def test_reader_positions_match_the_state(bset):
    cfg = bset.cfg
    s = bset.last_state()
    rd = D.ContactReader(cfg)
    imgs = render(s, cfg)
    ax, ay, found = rd.agent(imgs)
    gx, gy, gfound = rd.gem(imgs)
    assert found.all() and gfound.all()
    assert np.array_equal(ax, s.ax) and np.array_equal(ay, s.ay)
    assert np.array_equal(gx, s.gx) and np.array_equal(gy, s.gy)


def test_fast_uint8_path_equals_float_path(bset):
    cfg = bset.cfg
    imgs = render(bset.last_state(), cfg)
    rd = D.ContactReader(cfg)
    for color, size in ((cfg.color_agent, cfg.agent_size), (cfg.color_gem, cfg.gem_size), (cfg.color_enemy, cfg.enemy_size)):
        a, b = rd.locate(imgs, color, size), rd.locate(imgs.astype(np.float32) / 255.0, color, size)
        for u, v in zip(a, b):
            assert np.array_equal(u, v)
    c8, _ = rd.soft(imgs)
    cf, _ = rd.soft(imgs.astype(np.float32) / 255.0)
    assert np.allclose(c8, cf, atol=1e-5)


def test_agent_not_found_reads_no_contact(cfg):
    blank = np.zeros((3, cfg.size, cfg.size, 3), dtype=np.uint8)
    c, found = D.ContactReader(cfg).soft(blank)
    assert not found.any() and (c == 0).all()


def test_obstacle_on_the_rail_is_not_read_as_the_enemy(cfg):
    hx, hy = cfg.hub
    # agent sur le rail, au centre ; ennemi au bout du bras ; obstacle sous l'agent, sur le rail
    s = make_state(cfg, ax=hx + 1, ay=hy + 1, ox=hx, oy=hy, eo=3, ev=-1)
    c, found = D.ContactReader(cfg).soft(render(s, cfg))
    assert found.all() and abs(c[0]) < 0.02
    s2 = make_state(cfg, ax=hx + 1, ay=hy + 1, ox=hx, oy=hy, eo=0, ev=1)       # même scène, ennemi sur l'agent
    c2, _ = D.ContactReader(cfg).soft(render(s2, cfg))
    assert c2[0] > 0.98


def test_reader_handles_an_agent_straddling_two_rail_positions(cfg):
    ex, ey = enemy_xy(np.array([1]), cfg)
    s = make_state(cfg, ax=int(ex[0]) - 2, ay=int(ey[0]) + 1, eo=1, ev=1)       # à cheval sur les positions 0 et +1
    rd = D.ContactReader(cfg)
    assert rd.hard(render(s, cfg))[0][0] == 1.0
    s_far = make_state(cfg, ax=int(ex[0]) - 2, ay=int(ey[0]) + 1, eo=-2, ev=1)
    assert rd.hard(render(s_far, cfg))[0][0] == 0.0


def test_unknown_reader_is_rejected(cfg):
    with pytest.raises(ValueError, match="lecteur inconnu"):
        D.ContactReader(cfg).read(np.zeros((1, 64, 64, 3), dtype=np.uint8), "flou")


# ----------------------------------------------------------------------------- l'argument de linéarité (H4b)
def test_soft_reader_is_exact_on_the_mean_image_and_hard_reader_is_not(bset):
    cfg, p = bset.cfg, bset.cfg.p
    a, b, _ = bset.counterfactuals()
    mean = p * a.astype(np.float32) / 255.0 + (1 - p) * b.astype(np.float32) / 255.0
    t = D.true_costs(bset, 1)
    star, amb = t["cost"][:, 0], t["ambiguous"][:, 0]
    rd = D.ContactReader(cfg)
    soft, _ = rd.soft(mean)
    hard, _ = rd.hard(mean)
    assert np.abs(soft - star).max() < 0.01                         # lecteur doux : le bon risque, partout
    assert np.allclose(np.abs(hard - star)[amb], 0.3)               # lecteur à seuil : 0 ou 1, erreur min(p, 1-p)
    assert np.abs(hard - star)[~amb].max() == 0.0
    assert D.expected_error_mean(p, "hard") == pytest.approx(0.3) and D.expected_error_mean(p, "soft") == 0.0


# ----------------------------------------------------------------------------- chaîne complète sur prédicteurs factices
def test_expected_error_formula():
    assert D.expected_error_sampler(0.3, 0.3, 4) == pytest.approx(0.18522, abs=1e-5)
    assert D.expected_error_sampler(0.3, 0.3, 8) == pytest.approx(0.13341, abs=1e-5)
    assert D.expected_error_sampler(0.3, 0.3, 16) == pytest.approx(0.09182, abs=1e-5)
    rng = np.random.default_rng(0)
    mc = np.abs(rng.binomial(8, 0.188, 200_000) / 8 - 0.3).mean()
    assert D.expected_error_sampler(0.3, 0.188, 8) == pytest.approx(mc, abs=2e-3)
    assert D.expected_error_sampler(0.3, 0.3, 10_000) < 0.01        # beaucoup de tirages : l'erreur s'annule


def test_ideal_mean_rollout_has_no_soft_error_at_any_horizon(sub):
    res = D.evaluate_risk(D.BranchRollout(sub, 5), sub, (1, 3, 5), m=1)
    for k in ("1", "3", "5"):
        row = res["per_k"][k]
        assert row["soft"]["amb"]["mean"] < 0.01 and row["soft"]["other"]["mean"] < 0.01
        assert row["hard"]["amb"]["mean"] >= 0.29                    # le seuil se trompe d'au moins min(p, 1-p)
        assert row["agent_found"] == 1.0 and row["n_ambiguous"] > 20
    assert res["per_k"]["1"]["hard"]["amb"]["mean"] == pytest.approx(0.3, abs=1e-6)


def test_sharp_sampler_error_matches_theory_and_readers_agree(sub):
    res = D.evaluate_risk(D.BranchRollout(sub, 5, q=0.3), sub, (1, 5), m=8, seed=3)
    r1 = res["per_k"]["1"]
    n = r1["n_ambiguous"]
    assert r1["soft"]["amb"]["mean"] == pytest.approx(D.expected_error_sampler(0.3, 0.3, 8), abs=4 * 0.1 / np.sqrt(n))
    assert r1["soft"]["amb"]["mean"] == pytest.approx(r1["hard"]["amb"]["mean"], abs=0.005)   # images nettes
    assert r1["soft"]["other"]["mean"] < 0.01                        # jamais de fausse alerte
    assert abs(r1["soft"]["bias_amb"]) < 0.05                        # calibré : pas de biais
    biased = D.evaluate_risk(D.BranchRollout(sub, 5, q=0.0), sub, (1,), m=8)["per_k"]["1"]
    assert biased["soft"]["amb"]["mean"] > 0.25                      # effondrement sur la branche B : gros biais


def test_sampler_keeps_its_branch_along_a_rollout(sub):
    from flou.eval.rollout import autoregressive_rollout
    pred = D.BranchRollout(sub, 4, q=0.5)
    frames = autoregressive_rollout(pred, sub, 4, seed=1)
    paths = D.branch_paths(sub, 4)
    pick = pred._pick
    assert 0.3 < pick.mean() < 0.7
    for i in range(4):
        st = np.where(pick[:, None], paths["states_a"][:, i], paths["states_b"][:, i])
        assert np.array_equal(frames[:, i], render(State.from_array(st), sub.cfg))


def test_decision_instrument_report(bset):
    rep = DI.validate_decision(bset, (1, 3), m=4, planner=False)
    assert rep["h0e"]["pass"] and rep["h0f"] is None
    rows = rep["expected"]
    assert len(rows) == 3 and rows[0]["doux_k1"] < 0.01 and rows[0]["seuil_k1"] == pytest.approx(0.3, abs=1e-6)
    assert rows[1]["théorie_k1_seuil"] == pytest.approx(D.expected_error_sampler(0.3, 0.3, 4))
    assert 0.15 < DI.euler_q(0.3) < 0.25 and DI.euler_q(0.0) == 0.0  # biais d'Euler du cas jouet (≈ 0,19 à n = 3)
