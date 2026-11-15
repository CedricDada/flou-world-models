"""L'évaluateur doit reconnaître chacun des comportements factices dont on connaît la réponse."""
import numpy as np
import pytest

from flou.data.dataset import build_bifurcation_set
from flou.eval import baselines, fakes, rollout
from flou.eval import evaluate as ev
from flou.eval.classifier import CLASS_NAMES
from flou.game import GameConfig

RHO = 0.1


@pytest.fixture(scope="module")
def bset():
    return build_bifurcation_set(GameConfig().with_p(0.3), n=1000, seed=31)


@pytest.fixture(scope="module")
def results(bset):
    preds = {
        "ideal": fakes.IdealSampler(bset),
        "biased": fakes.IdealSampler(bset, q=0.5, name="biased"),
        "mean": fakes.IdealMean(bset),
        "collapse": fakes.ModeCollapse(bset),
        "oracle": fakes.OracleRegressor(bset),
        "copy": baselines.CopyLast(),
        "dropgem": fakes.DropGem(bset),
        "noisy": fakes.Noisy(fakes.IdealSampler(bset), 0.02),
    }
    return {k: ev.evaluate_predictor(v, bset, RHO, seed=5) for k, v in preds.items()}


def frac(r, name):
    return r["counts"][name] / r["n"]


def test_ideal_sampler_is_sharp_and_calibrated(results):
    r = results["ideal"]
    assert frac(r, "floue") == 0 and frac(r, "autre") == 0
    assert r["p_hat"]["ci_lo"] <= 0.3 <= r["p_hat"]["ci_hi"] and r["p_abs_err"] < 0.05
    assert r["recall"]["value"] == 1.0 and r["hallucination"]["value"] == 0.0
    assert r["off_palette"] == 0.0


def test_ideal_regressor_is_fully_blurred_and_at_the_bayes_floor(results, bset):
    r = results["mean"]
    assert frac(r, "floue") == 1.0                       # p = 0,3 est dans ]0,2 ; 0,8[
    assert r["alpha_mean"] == pytest.approx(0.3, abs=1e-6)
    assert r["mse"]["ratio_expected"] == pytest.approx(1.0, abs=1e-6)       # plancher atteint exactement
    assert 0.9 < r["mse"]["ratio_realized"] < 1.1                            # bruit d'étiquette seulement
    assert r["p_hat"]["n"] == 0 and np.isnan(r["p_hat"]["value"])           # aucune image nette : p̂ indéfini
    assert r["off_palette"] > 0                                              # les mélanges sortent de la palette
    chk = ev.check_h0b(r)
    assert chk["pass"] and chk["fit_ok"] and not chk["leak"]


def test_a_perfect_sampler_pays_twice_the_floor(results):
    # un échantillonneur parfait a une MSE de 2 x plancher (variance du futur + variance de l'échantillon)
    assert results["ideal"]["mse"]["ratio_expected"] == pytest.approx(2.0, abs=0.15)


def test_mode_collapse_is_sharp_but_never_calibrated(results):
    r = results["collapse"]
    assert frac(r, "floue") == 0 and frac(r, "autre") == 0
    assert r["p_hat"]["value"] == 0.0 and r["p_abs_err"] == pytest.approx(0.3)
    assert not r["p_hat_covers_p"]


def test_biased_sampler_is_detected_by_the_confidence_interval(results):
    r = results["biased"]
    assert frac(r, "floue") == 0
    assert not r["p_hat_covers_p"] and r["p_abs_err"] > 0.15


def test_oracle_regressor_is_sharp_and_leaks_only_in_realized_mse(results):
    r = results["oracle"]
    assert frac(r, "floue") == 0                                           # H0c : moins de 5 % de flou
    assert ev.check_blur_below(r)["pass"]
    assert r["mse"]["sq_err_realized"] == pytest.approx(0.0, abs=1e-9)    # connaît la branche tirée
    chk = ev.check_h0b(r)
    assert chk["leak"] and not chk["pass"]                                 # ... ce que H0b doit détecter


def test_h0b_detects_underfitting(bset):
    class Blurrier(fakes.IdealMean):
        def predict(self, batch, rng):
            y = super().predict(batch, rng)
            return 0.7 * y + 0.3 * y.mean()          # détail perdu : au-dessus du plancher
    r = ev.evaluate_predictor(Blurrier(bset), bset, RHO)
    chk = ev.check_h0b(r)
    assert r["mse"]["ratio_expected"] > 1.10 and not chk["fit_ok"] and not chk["pass"]


def test_copy_baseline_sets_the_scale(results):
    r = results["copy"]
    assert frac(r, "autre") == 1.0
    assert r["mse"]["ratio_expected"] > 3 * results["ideal"]["mse"]["ratio_expected"]


def test_dropgem_is_caught_by_recall_and_by_the_classifier(results):
    r = results["dropgem"]
    assert r["recall"]["value"] == 0.0 and r["recall"]["ci_hi"] < 0.01
    assert frac(r, "autre") == 1.0                        # rho_broken ≈ 0,33 > ρmax


def test_noise_above_rho_max_turns_everything_into_other(results):
    assert frac(results["noisy"], "autre") > 0.99         # ρmax doit suivre le niveau de bruit du générateur


def test_summary_is_json_serialisable_and_per_sample_is_aligned(results):
    import json
    r = results["ideal"]
    s = ev.summarize(r)
    json.dumps(s)
    assert "per_sample" not in s
    assert len(r["per_sample"]["label"]) == r["n"] == len(r["per_sample"]["state"])
    assert set(r["counts"]) == set(CLASS_NAMES)


def test_repeated_sampling_pools_and_is_reproducible(bset):
    sub = bset.subset(np.arange(400))
    p = fakes.IdealSampler(sub)
    r1 = ev.evaluate_predictor(p, sub, RHO, n_samples=3, seed=9)
    r2 = ev.evaluate_predictor(p, sub, RHO, n_samples=3, seed=9)
    assert r1["n"] == 3 * len(sub) and r1["counts"] == r2["counts"]
    single = ev.evaluate_predictor(p, sub, RHO, n_samples=1, seed=9)
    width = lambda r: r["p_hat"]["ci_hi"] - r["p_hat"]["ci_lo"]
    assert width(r1) < width(single)                                  # plus d'échantillons, intervalle plus étroit
    assert ev.evaluate_predictor(p, sub, RHO, n_samples=3, seed=10)["counts"] != r1["counts"]


def test_calibration_interval_has_nominal_coverage(bset):
    """Un générateur parfait doit être couvert par l'IC à 95 % dans la grande majorité des tirages
    (la couverture exacte de Clopper-Pearson est testée dans test_eval_stats)."""
    sub = bset.subset(np.arange(250))
    p = fakes.IdealSampler(sub)
    hits = sum(ev.evaluate_predictor(p, sub, RHO, seed=s)["p_hat_covers_p"] for s in range(30))
    assert hits >= 26


def test_bayes_floor_values():
    assert ev.bayes_floor(np.array([2.0]), 0.5)[0] == pytest.approx(1.0)
    assert [round(ev.bayes_floor(np.array([1.0]), p)[0], 2) for p in (0.1, 0.3, 0.5)] == [0.09, 0.21, 0.25]


# ---------------------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------------------
def test_only_the_enemy_depends_on_the_branch_sequence(bset):
    t1, s1 = rollout.rollout_truth(bset, 16, seed=1)
    t2, s2 = rollout.rollout_truth(bset, 16, seed=2)
    for a, b in zip(s1, s2):
        for f in ("ax", "ay", "ox", "oy", "gx", "gy"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert any(not np.array_equal(a.eo, b.eo) for a, b in zip(s1, s2))
    assert not np.array_equal(t1, t2)                      # les images diffèrent par l'ennemi seulement
    assert np.array_equal(s1[0].eo, s2[0].eo)              # le pas 0 utilise la branche du jeu de test


def test_rollout_truth_first_frame_is_the_realized_next_image(bset):
    truth, _ = rollout.rollout_truth(bset, 4, seed=0)
    assert np.array_equal(truth[:, 0], bset.realized_images())


def test_rollout_horizon_validation(bset):
    with pytest.raises(ValueError):
        rollout.rollout_truth(bset, bset.horizon + 1)
    assert bset.horizon == 16 and bset.future_actions.shape == (len(bset), 16)
    assert np.array_equal(bset.future_actions[:, 0], bset.actions)


def test_truth_rollout_is_perfect_and_drift_is_detected(bset):
    sub = bset.subset(np.arange(200))
    truth, _ = rollout.rollout_truth(sub, 16, seed=1)
    ok = rollout.evaluate_rollout(fakes.TruthRollout(truth), sub, 16, seed=1)
    assert min(ok["track_ok"]) > 0.99 and min(ok["gem_recall"]) == 1.0
    assert max(ok["mse_static"]) == 0.0 and max(ok["off_palette"]) == 0.0
    bad = rollout.evaluate_rollout(fakes.TruthRollout(truth, drift_after=8), sub, 16, seed=1)
    assert min(bad["track_ok"][:8]) > 0.99 and max(bad["track_ok"][8:]) == 0.0
    assert bad["quality_last"] == 0.0 and max(bad["mse_static"][8:]) > 0


def test_copy_baseline_loses_track_of_the_agent(bset):
    sub = bset.subset(np.arange(200))
    r = rollout.evaluate_rollout(baselines.CopyLast(), sub, 16, seed=1)
    assert r["track_ok"][-1] < 0.1 and r["mse_static"][-1] > r["mse_static"][0]


def test_subset_keeps_arrays_aligned(bset):
    sub = bset.subset(np.array([5, 7, 11]))
    assert len(sub) == 3 and sub.future_actions.shape == (3, 16)
    assert np.array_equal(sub.actions, bset.actions[[5, 7, 11]])
    assert np.array_equal(sub.counterfactuals()[0], bset.counterfactuals()[0][[5, 7, 11]])


def test_bifurcation_set_roundtrip_keeps_future_actions(bset, tmp_path):
    path = tmp_path / "t.npz"
    bset.save(path, {"config": bset.cfg.to_dict()})
    again = type(bset).load(path)
    assert np.array_equal(again.future_actions, bset.future_actions)
    assert again.horizon == 16
