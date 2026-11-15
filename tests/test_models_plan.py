"""Plans d'architecture, comptage de paramètres et mathématiques d'EDM : tout en NumPy, sans PyTorch."""
import numpy as np
import pytest

from flou.models import edm_math as E
from flou.models.config import (GPTConfig, UNetConfig, VQConfig, check_plan, unet_plan, vq_plan)
from flou.models import params as PR


# ---------------------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("channels", [(48, 64, 96, 128), (64, 64, 64, 64), (32, 64, 128, 128), (32, 64)])
@pytest.mark.parametrize("extra", [0, 2])
def test_unet_plan_is_consistent(channels, extra):
    cfg = UNetConfig(in_ch=15, channels=channels, n_extra_cond=extra)
    plan = unet_plan(cfg)
    check_plan(plan, cfg)
    kinds = [op[0] for op in plan]
    n_lvl = len(channels)
    assert kinds.count("res_enc") == n_lvl * cfg.n_res
    assert kinds.count("res_dec") == n_lvl * (cfg.n_res + 1)
    assert kinds.count("down") == kinds.count("up") == n_lvl - 1
    assert plan[0][1] == 15 + extra and plan[-1][2] == 3


def test_check_plan_catches_a_broken_skip():
    cfg = UNetConfig()
    plan = unet_plan(cfg)
    i = next(i for i, op in enumerate(plan) if op[0] == "res_dec")
    bad = list(plan)
    bad[i] = ("res_dec", plan[i][1], plan[i][2], plan[i][3] + 1)
    with pytest.raises(AssertionError):
        check_plan(bad, cfg)


def test_unet_config_validation():
    with pytest.raises(ValueError):
        UNetConfig(channels=(48, 60, 96, 128), groups=8)      # 60 non divisible par 8
    with pytest.raises(ValueError):
        UNetConfig(channels=(32,) * 8)                         # trop de niveaux pour 64 pixels


def test_param_counts_are_in_the_budget_of_the_course():
    a = PR.unet_params(UNetConfig(in_ch=12))                                     # A : régression
    b = PR.unet_params(UNetConfig(in_ch=15, sigma_cond=True))                    # B : débruiteur EDM
    o = PR.unet_params(UNetConfig(in_ch=12, n_extra_cond=2))                     # oracle
    c = PR.vqvae_params(VQConfig(K=64)), PR.vqvae_params(VQConfig(K=1024))
    g = PR.gpt_params(GPTConfig(K=64)), PR.gpt_params(GPTConfig(K=1024))
    for n in (a, b, o, c[0] + g[0], c[1] + g[1]):
        assert 4e6 <= n <= 8e6, n
    # budget apparié : même ordre de grandeur, A et B presque identiques
    assert abs(a - b) / a < 0.03 and abs(o - a) / a < 0.01
    assert max(a, b, c[1] + g[1]) / min(a, b, c[0] + g[0]) < 1.25


def test_param_formula_hand_checks():
    assert PR.conv(3, 8, 3) == 3 * 8 * 9 + 8
    assert PR.res_block(64, 64, 256) == 2 * 64 + (64 * 64 * 9 + 64) + (256 * 128 + 128) + (64 * 64 * 9 + 64)
    assert PR.res_block(32, 64, 256) - PR.res_block(64, 64, 256) < 0                  # moins de canaux : moins de poids,
    # ... mais le saut 1x1 s'ajoute quand cin != cout :
    assert PR.res_block(32, 64, 16) > PR.groupnorm(32) + PR.conv(32, 64, 3) + PR.conv(64, 64, 3)


def test_unet_cost_is_reasonable():
    macs = PR.unet_macs(UNetConfig(in_ch=15, sigma_cond=True))
    assert 1e9 < macs < 4e9          # ~2,5 GMAC par exemple : ordre de grandeur pour une T4


# ---------------------------------------------------------------------------------------
# VQ-VAE et GPT
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("grid,down", [(4, 4), (8, 3), (16, 2), (32, 1)])
def test_vq_plan_downsampling_matches_grid(grid, down):
    cfg = VQConfig(grid=grid)
    assert cfg.n_down == down and cfg.n_tokens == grid * grid
    plan = vq_plan(cfg)
    assert sum(op[0] == "enc_down" for op in plan) == down == sum(op[0] == "dec_up" for op in plan)
    assert 64 // 2 ** down == grid


def test_vq_config_validation():
    with pytest.raises(ValueError):
        VQConfig(grid=6)
    with pytest.raises(ValueError):
        VQConfig(grid=64)          # aucun sous-échantillonnage : refusé


def test_gpt_sequence_and_bits():
    g = GPTConfig(K=256, n_tokens=64, ctx_frames=4)
    assert g.seq_len == 4 * 64 + 1 + 64 == 321
    assert 64 * np.log2(256) == 512 and 64 * np.log2(64) == 384 and 64 * np.log2(1024) == 640   # chap. 4
    with pytest.raises(ValueError):
        GPTConfig(d=100, n_head=3)


# ---------------------------------------------------------------------------------------
# EDM
# ---------------------------------------------------------------------------------------
def test_karras_schedule():
    s = E.karras_sigmas(3)
    assert len(s) == 4 and s[0] == 80.0 and s[-2] == pytest.approx(0.002) and s[-1] == 0.0
    assert np.all(np.diff(s) < 0)
    assert list(E.karras_sigmas(1)) == [80.0, 0.0]
    assert len(E.karras_sigmas(20)) == 21
    with pytest.raises(ValueError):
        E.karras_sigmas(0)


def test_edm_preconditioning_identities():
    sig = np.exp(np.linspace(-6, 4, 50))
    cs, co, ci = E.precond_coefficients(sig)
    sd = E.SIGMA_DATA
    # entrée de variance 1 pour tout sigma
    assert np.allclose(ci ** 2 * (sig ** 2 + sd ** 2), 1.0)
    # la cible effective de F a variance 1 : c_out^2 = (1 - c_skip)^2 sd^2 + c_skip^2 sig^2
    assert np.allclose(co ** 2, (1 - cs) ** 2 * sd ** 2 + cs ** 2 * sig ** 2)
    # la pondération rend l'erreur de F de poids 1 : λ c_out² = 1
    assert np.allclose(E.loss_weight(sig) * co ** 2, 1.0)
    # limites : σ → 0, D = x (sortie = entrée) ; σ → ∞, D ne dépend plus de x
    assert E.precond_coefficients(1e-6)[0] == pytest.approx(1.0, abs=1e-6)
    assert E.precond_coefficients(1e6)[0] == pytest.approx(0.0, abs=1e-6)


def test_noise_sampling_distribution():
    s = E.sample_sigma(np.random.default_rng(0), 200_000)
    assert np.log(s).mean() == pytest.approx(-1.2, abs=0.01) and np.log(s).std() == pytest.approx(1.2, abs=0.01)


def test_exact_denoiser_is_the_posterior_mean():
    rng = np.random.default_rng(1)
    p, sig = 0.3, 0.7
    y = (rng.random(400_000) < p).astype(float)
    x = y + sig * rng.standard_normal(y.size)
    d = E.exact_denoiser(x, sig, p)
    # E[y | x] : la régression de y sur d(x) a pour pente 1 (calibration du débruiteur)
    assert np.polyfit(d, y, 1)[0] == pytest.approx(1.0, abs=0.03)
    assert abs((d - y).mean()) < 0.01


def test_one_step_euler_collapses_to_the_mean_and_three_steps_are_sharp_but_biased():
    """Résultats du cours (§5.7), recalculés."""
    x1 = E.euler_sample_toy(0.3, 1, 100_000)
    assert np.abs(x1 - 0.3).max() < 0.02                           # n = 1 : tout vaut p -> 100 % de flou
    assert E.toy_p_hat(x1)[1] > 0.999
    x3 = E.euler_sample_toy(0.3, 3, 200_000)
    p3, blur3 = E.toy_p_hat(x3)
    assert blur3 < 1e-3 and p3 == pytest.approx(0.015, abs=0.01)    # net, mais presque tout sur la branche majoritaire
    x3b = E.euler_sample_toy(0.3, 3, 200_000, sigma_max=5.0)
    assert E.toy_p_hat(x3b)[0] == pytest.approx(0.19, abs=0.02)


def test_euler_calibration_improves_slowly_with_n():
    ph = {n: E.toy_p_hat(E.euler_sample_toy(0.3, n, 200_000))[0] for n in (5, 10, 20, 50)}
    assert ph[5] < ph[10] < ph[20] < ph[50] < 0.3 + 0.01
    assert ph[10] == pytest.approx(0.26, abs=0.02) and ph[50] == pytest.approx(0.29, abs=0.015)
    # à p = 0,5, le biais disparaît par symétrie dès n = 2 : cela peut masquer le phénomène
    assert E.toy_p_hat(E.euler_sample_toy(0.5, 2, 200_000))[0] == pytest.approx(0.5, abs=0.01)
