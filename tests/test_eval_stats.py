"""Les nombres du cours (chap. 7) sont recalculés, jamais recopiés."""
import math

import numpy as np
import pytest

from flou.eval import stats as S


def test_three_seed_thresholds_match_course():
    assert S.unpaired_threshold(3, 3) == pytest.approx(2.27, abs=0.005)
    assert S.mean_ci_halfwidth(3) == pytest.approx(2.48, abs=0.005)
    assert S.paired_threshold(3) == pytest.approx(2.48, abs=0.005)


def test_bonferroni_thresholds_match_course():
    assert [round(S.unpaired_threshold(k_tests=k), 2) for k in (3, 6, 10)] == [3.23, 3.96, 4.57]
    assert [round(S.paired_threshold(k_tests=k), 2) for k in (3, 6, 10)] == [4.42, 6.28, 8.13]


def test_power_matches_course():
    got = [S.power_unpaired(d) for d in (1, 2, 2.3, 3)]
    assert got == pytest.approx([0.16, 0.46, 0.57, 0.78], abs=0.01)


def test_family_wise_error():
    assert S.family_wise_error(10) == pytest.approx(0.40, abs=0.005)
    assert S.family_wise_error(30) == pytest.approx(0.79, abs=0.005)


def test_binomial_table_and_required_n():
    assert S.wald_half_width(0.5, 1000) == pytest.approx(0.031, abs=0.0005)
    assert S.wald_half_width(0.5, 5000) == pytest.approx(0.014, abs=0.0005)
    assert S.wald_half_width(0.3, 1000) == pytest.approx(0.028, abs=0.0005)
    assert S.wald_half_width(0.1, 1000) == pytest.approx(0.019, abs=0.0005)
    assert S.required_n(0.5, 0.02) == 2401
    assert S.required_n(0.3, 0.02) == 2017
    assert S.prob_error_exceeds(0.5, 1000, 0.05) == pytest.approx(0.0016, abs=0.0002)


def test_clopper_pearson_known_values_and_edges():
    lo, hi = S.clopper_pearson(15, 20)
    assert (round(lo, 2), round(hi, 2)) == (0.51, 0.91)          # exemple du cours
    assert S.clopper_pearson(0, 50)[0] == 0.0 and S.clopper_pearson(50, 50)[1] == 1.0
    lo, hi = S.clopper_pearson(0, 1000)
    assert hi == pytest.approx(1 - 0.025 ** (1 / 1000), rel=1e-6)   # forme fermée pour k = 0
    assert math.isnan(S.clopper_pearson(0, 0)[0])
    with pytest.raises(ValueError):
        S.clopper_pearson(5, 3)


def test_clopper_pearson_coverage_is_at_least_nominal():
    rng = np.random.default_rng(0)
    for p in (0.1, 0.5):
        k = rng.binomial(200, p, 3000)
        cover = np.mean([S.clopper_pearson(int(ki), 200)[0] <= p <= S.clopper_pearson(int(ki), 200)[1] for ki in k])
        assert cover >= 0.94   # l'intervalle exact est conservateur (≥ 95 % en théorie, bruit de simulation)


def test_wilson_close_to_exact_for_large_n():
    a, b = S.wilson_ci(300, 1000), S.clopper_pearson(300, 1000)
    assert a[0] == pytest.approx(b[0], abs=0.003) and a[1] == pytest.approx(b[1], abs=0.003)


def test_p_hat_uses_sharp_images_only():
    r = S.p_hat(30, 70)
    assert r.value == pytest.approx(0.3) and r.n == 100 and r.lo < 0.3 < r.hi
    assert math.isnan(S.p_hat(0, 0).value)


def test_compare_groups_unpaired_threshold_is_used():
    s = 1.0
    base = np.array([-s, 0.0, s])                         # écart-type exactement 1
    # décalage juste sous / juste au-dessus de 2,27 s
    below = S.compare_groups(base + 2.20, base)
    above = S.compare_groups(base + 2.35, base)
    assert not below.significant and below.verdict == "non concluant"
    assert above.significant and above.verdict == "concluant"
    assert below.threshold == pytest.approx(2.267, abs=1e-3) and below.sd == pytest.approx(1.0)
    assert above.diff == pytest.approx(2.35)


def test_compare_groups_matches_scipy_t_test():
    from scipy import stats
    a, b = [0.31, 0.35, 0.29], [0.52, 0.47, 0.55]
    r = S.compare_groups(a, b)
    ref = stats.ttest_ind(a, b)
    assert r.t == pytest.approx(ref.statistic, rel=1e-9) and r.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    rp = S.compare_groups(a, b, paired=True)
    refp = stats.ttest_rel(a, b)
    assert rp.t == pytest.approx(refp.statistic, rel=1e-9) and rp.p_value == pytest.approx(refp.pvalue, rel=1e-9)
    # la décision à seuil et la décision à p-value s'accordent
    assert r.significant == (r.p_value < 0.05) and rp.significant == (rp.p_value < 0.05)


def test_compare_groups_validation_and_degenerate_cases():
    with pytest.raises(ValueError):
        S.compare_groups([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        S.compare_groups([1, 2, 3], [1, 2], paired=True)
    same = S.compare_groups([1, 1, 1], [1, 1, 1])
    assert not same.significant and same.p_value == 1.0
    sep = S.compare_groups([2, 2, 2], [1, 1, 1])                 # variance nulle, écart non nul
    assert sep.significant and sep.p_value == 0.0


def test_false_positive_rate_of_unpaired_rule_is_five_percent():
    rng = np.random.default_rng(1)
    hits = 0
    trials = 4000
    for _ in range(trials):
        a, b = rng.normal(size=3), rng.normal(size=3)
        hits += S.compare_groups(a, b).significant
    assert hits / trials == pytest.approx(0.05, abs=0.015)


def test_power_is_consistent_with_simulation():
    rng = np.random.default_rng(2)
    d, trials = 2.0, 4000
    hits = sum(S.compare_groups(rng.normal(size=3) + d, rng.normal(size=3)).significant for _ in range(trials))
    assert hits / trials == pytest.approx(S.power_unpaired(d), abs=0.03)


def test_mean_ci_covers_the_true_mean():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(3000):
        m, lo, hi = S.mean_ci(rng.normal(5, 2, 3))
        hits += lo <= 5 <= hi
    assert hits / 3000 == pytest.approx(0.95, abs=0.015)


def test_cohen_kappa():
    assert S.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert S.cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    # exemple chiffré : po = 0.7, pe = 0.5 -> κ = 0.4
    r1 = [1] * 5 + [0] * 5
    r2 = [1] * 4 + [0] + [0] * 3 + [1] * 2
    assert S.cohen_kappa(r1, r2) == pytest.approx((0.7 - 0.5) / 0.5)
    assert math.isnan(S.cohen_kappa([1, 1, 1], [1, 1, 1]))      # pe = 1 : indéfini
    with pytest.raises(ValueError):
        S.cohen_kappa([1, 2], [1])


def test_mean_pairwise_kappa_perfect_agreement():
    r = np.array([[0, 1, 1, 0, 1], [0, 1, 1, 0, 1], [0, 1, 1, 0, 1]])
    assert S.mean_pairwise_kappa(r) == pytest.approx(1.0)
