"""Classifieur contrefactuel, contrôle H0a, sonde analytique, rappel et artefacts."""
import numpy as np
import pytest

from flou.data.dataset import build_bifurcation_set
from flou.eval import classifier as C
from flou.eval import probe as P
from flou.eval import recall as R
from flou.game import GameConfig
from flou.game.render import render


@pytest.fixture(scope="module")
def bset():
    return build_bifurcation_set(GameConfig(), n=300, seed=21)


@pytest.fixture(scope="module")
def cf(bset):
    a, b, d = bset.counterfactuals()
    return a, b, d


def test_alpha_rho_course_examples(cf):
    a, b, _ = cf
    fa, fb = a[0] / 255.0, b[0] / 255.0
    al, rho = C.alpha_rho(0.3 * fa + 0.7 * fb, a[0], b[0])
    assert al == pytest.approx(0.3) and rho == pytest.approx(0.0, abs=1e-9)
    assert C.alpha_rho(a[0], a[0], b[0]) == (pytest.approx(1.0), pytest.approx(0.0, abs=1e-9))
    assert C.alpha_rho(b[0], a[0], b[0]) == (pytest.approx(0.0), pytest.approx(0.0, abs=1e-9))


def test_alpha_rho_batch_equals_single(cf):
    a, b, _ = cf
    rng = np.random.default_rng(0)
    y = np.clip(a[:20] / 255.0 * 0.6 + b[:20] / 255.0 * 0.4 + rng.normal(0, 0.02, a[:20].shape), 0, 1)
    al, rho = C.alpha_rho(y, a[:20], b[:20])
    for i in range(20):
        ai, ri = C.alpha_rho(y[i], a[i], b[i])
        assert al[i] == pytest.approx(ai) and rho[i] == pytest.approx(ri)


def test_alpha_clipped_and_rho_is_distance_to_segment():
    A = np.zeros(10); A[0] = 1.0                  # δ = 1
    B = np.zeros(10)
    # au-delà de A sur la droite (A, B) : α̂ borné à 1, ρ = distance à A
    al, rho = C.alpha_rho(2.0 * A, A, B)
    assert al == 1.0 and rho == pytest.approx(1.0)
    # point orthogonal au segment, à mi-chemin : α̂ = 0,5 ; ρ = hauteur
    y = np.zeros(10); y[0] = 0.5; y[1] = 0.3
    al, rho = C.alpha_rho(y, A, B)
    assert al == pytest.approx(0.5) and rho == pytest.approx(0.3)


def test_no_bifurcation_gives_nan_and_other():
    x = np.random.default_rng(0).random((2, 5))
    al, rho = C.alpha_rho(x, x, x)
    assert np.isnan(al).all() and np.isnan(rho).all()
    assert (C.classify(al, rho, 0.2) == C.CLASS_OTHER).all()


def test_classify_thresholds_and_ideal_class():
    eps = 0.2
    al = np.array([1.0, 0.8, 0.79, 0.5, 0.21, 0.2, 0.0, 0.5])
    rho = np.array([0, 0, 0, 0, 0, 0, 0, 0.5])
    lab = C.classify(al, rho, rho_max=0.2, eps=eps)
    assert list(lab) == [C.CLASS_A, C.CLASS_A, C.CLASS_BLUR, C.CLASS_BLUR, C.CLASS_BLUR, C.CLASS_B,
                         C.CLASS_B, C.CLASS_OTHER]
    # question 2.1 du cours : p = 0,3 -> flou ; p = 0,15 -> nette-B ; p = 0,8 -> nette-A (α̂ = 0,8 est net)
    assert C.ideal_class(0.3) == C.CLASS_BLUR and C.ideal_class(0.15) == C.CLASS_B
    assert C.ideal_class(0.8) == C.CLASS_A and C.ideal_class(0.5) == C.CLASS_BLUR


def test_ideal_mixture_is_classified_blur_exactly_for_p_between_eps(cf):
    a, b, _ = cf
    for p, expected in [(0.1, C.CLASS_B), (0.19, C.CLASS_B), (0.3, C.CLASS_BLUR), (0.5, C.CLASS_BLUR),
                        (0.7, C.CLASS_BLUR), (0.81, C.CLASS_A), (0.9, C.CLASS_A)]:
        y = p * a / 255.0 + (1 - p) * b / 255.0
        res = C.classify_images(y, a, b, rho_max=0.05)
        assert (res["label"] == expected).all(), p
        assert res["alpha"].mean() == pytest.approx(p, abs=1e-6)


def test_class_counts_fractions():
    lab = np.array([0, 0, 1, 2, 3, 3, 3, 3])
    assert C.class_counts(lab) == {"nette-A": 2, "nette-B": 1, "floue": 1, "autre": 4}
    assert C.class_fractions(lab)["autre"] == 0.5


def test_h0a_passes_without_noise_and_flags_destroyed_gem(bset, cf):
    a, b, _ = cf
    s_a, _ = bset.next_states()
    gem = np.stack([s_a.gx, s_a.gy], axis=1)
    r = C.h0a_check(a, b, gem, rho_max=0.1, noise_sigma=0.0)
    assert r["pass"] and r["alpha_mae_mix"] < 1e-6 and r["pure_ok"] == 1.0
    assert r["broken_flagged_other"] == 1.0                 # un petit objet détruit est « autre »
    assert r["rho_broken_min"] > 0.3                        # ≈ 2,5 / δ : très au-dessus de ρmax = 0,1


def test_tuned_rho_max_gives_99_percent_on_noisy_pure_images(bset, cf):
    a, b, _ = cf
    rm = C.tune_rho_max(a, b, 0.005, np.random.default_rng(0), quantile=0.995)
    r = C.h0a_check(a, b, None, rho_max=rm, noise_sigma=0.005, seed=3)
    assert r["pure_ok"] > 0.99 and r["alpha_mae_mix"] < 0.05 and r["pass"]
    # une valeur trop petite échoue : le contrôle est capable d'échouer
    bad = C.h0a_check(a, b, None, rho_max=rm / 4, noise_sigma=0.005, seed=3)
    assert not bad["pass_pure"] and not bad["pass"]


def test_rho_max_has_to_follow_the_noise_level(cf):
    a, b, _ = cf
    rng = np.random.default_rng(0)
    r5, r20 = (C.tune_rho_max(a, b, s, rng) for s in (0.005, 0.02))
    assert r20 > 3 * r5 and r5 == pytest.approx(0.08, abs=0.02)   # ρ ≈ σ·sqrt(12288)/δ


# ---------------------------------------------------------------------------------------
# Sonde analytique
# ---------------------------------------------------------------------------------------
def test_analytic_probe_exact_on_isolated_objects(cfg):
    from conftest import make_state
    s = make_state(cfg, ax=20, ay=22, ox=40, oy=40, eo=-2, gx=6, gy=56)
    img = render(s, cfg)
    pred = P.AnalyticProbe(cfg)(img)
    truth = P.true_positions(s, cfg)
    assert np.allclose(pred, truth, atol=1e-9)


def test_analytic_probe_meets_the_course_criterion_on_real_states(bset):
    s = bset.last_state()
    rep = P.probe_report(P.AnalyticProbe(bset.cfg)(render(s, bset.cfg)), P.true_positions(s, bset.cfg))
    for name in ("agent", "obstacle", "enemy", "gem"):
        assert rep[name]["found"] > 0.99 and rep[name]["rmse"] < 0.5


def test_analytic_probe_finds_nothing_in_a_blurred_enemy(bset, cf):
    a, b, _ = cf
    blur = (0.5 * a.astype(np.float64) + 0.5 * b.astype(np.float64)) / 255.0
    pos = P.AnalyticProbe(bset.cfg)(blur)
    assert np.isnan(pos[:, 2]).all()           # ni l'une ni l'autre position nette de l'ennemi
    assert not np.isnan(pos[:, 0]).any()       # l'agent, lui, est net


# ---------------------------------------------------------------------------------------
# Rappel et artefacts
# ---------------------------------------------------------------------------------------
def test_gem_recall_hallucination_and_partial(bset, cf):
    cfg = bset.cfg
    a, _, _ = cf
    s_a, _ = bset.next_states()
    gx, gy = s_a.gx, s_a.gy
    assert R.gem_recall(a, gx, gy, cfg).all() and not R.gem_hallucination(a, gx, gy, cfg).any()
    gone = a.copy()
    bg = np.asarray(cfg.color_bg, dtype=np.uint8)
    for i in range(len(gone)):
        gone[i, gy[i]:gy[i] + 2, gx[i]:gx[i] + 2] = bg
    assert not R.gem_recall(gone, gx, gy, cfg).any()
    # un seul pixel sur quatre : sous le seuil de 2 pixels ; deux pixels : présent
    one = gone.copy(); two = gone.copy()
    gem = np.asarray(cfg.color_gem, dtype=np.uint8)
    for i in range(len(one)):
        one[i, gy[i], gx[i]] = gem
        two[i, gy[i], gx[i]] = gem
        two[i, gy[i], gx[i] + 1] = gem
    assert not R.gem_recall(one, gx, gy, cfg).any() and R.gem_recall(two, gx, gy, cfg).all()
    # objet déplacé : absent à sa place et halluciné ailleurs
    moved = gone.copy()
    for i in range(len(moved)):
        x2 = 30 if gx[i] < 20 else 8
        moved[i, 8:10, x2:x2 + 2] = gem
    assert not R.gem_recall(moved, gx, gy, cfg).any() and R.gem_hallucination(moved, gx, gy, cfg).all()


def test_half_blended_gem_is_not_recalled(bset, cf):
    cfg = bset.cfg
    a, b, _ = cf
    s_a, _ = bset.next_states()
    bg = np.asarray(cfg.color_bg, dtype=np.float64)
    gem = np.asarray(cfg.color_gem, dtype=np.float64)
    faded = a.astype(np.float64).copy()
    for i in range(len(faded)):
        faded[i, s_a.gy[i]:s_a.gy[i] + 2, s_a.gx[i]:s_a.gx[i] + 2] = 0.5 * bg + 0.5 * gem
    assert not R.gem_recall(faded / 255.0, s_a.gx, s_a.gy, cfg).any()


def test_off_palette_zero_on_real_images_and_grows_with_blur(bset, cf):
    cfg = bset.cfg
    a, b, _ = cf
    assert (R.off_palette_fraction(a, cfg) == 0).all()
    mix = (0.5 * a.astype(np.float64) + 0.5 * b.astype(np.float64)) / 255.0
    assert (R.off_palette_fraction(mix, cfg) > 0).all()
    # le même résultat sur uint8 et sur flottants quantifiés
    q = np.round(mix * 255).astype(np.uint8)
    assert np.allclose(R.off_palette_fraction(q, cfg), R.off_palette_fraction(q / 255.0, cfg))


def test_block_edge_excess_detects_block_artifacts():
    rng = np.random.default_rng(0)
    smooth = np.broadcast_to(np.linspace(0, 1, 64)[None, :, None, None], (4, 64, 64, 3)).copy()
    assert np.abs(R.block_edge_excess(smooth, 8)).max() < 1e-9      # pas de marche aux frontières
    blocky = np.kron(rng.random((4, 8, 8, 3)), np.ones((1, 8, 8, 1)))
    assert (R.block_edge_excess(blocky, 8) > 0.05).all()            # marches aux frontières de blocs
    assert (R.block_edge_excess(blocky, 8) > R.block_edge_excess(smooth, 8)).all()
