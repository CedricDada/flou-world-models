import numpy as np
from conftest import make_state

from flou.game import (
    BRANCH_A, BRANCH_B, NOOP, advance, branch_delta, enemy_xy, initial_state,
    is_branch_state, render, render_counterfactual,
)


def count(img, color):
    return int((img == np.asarray(color, dtype=np.uint8)).all(axis=-1).sum())


def test_shape_dtype_and_purity(cfg):
    s = initial_state(cfg, 5, np.random.default_rng(0))
    img = render(s, cfg)
    assert img.shape == (5, 64, 64, 3) and img.dtype == np.uint8
    assert np.array_equal(img, render(s.copy(), cfg))      # fonction pure de l'état
    assert np.array_equal(render(s[2], cfg)[0], img[2])    # lot ou élément isolé : même image


def test_each_object_has_its_exact_size_and_color(cfg):
    # objets séparés : aucun recouvrement
    s = make_state(cfg, ax=30, ay=10, ox=40, oy=45, eo=0, ev=-1, gx=6, gy=56)
    img = render(s, cfg)[0]
    assert count(img, cfg.color_agent) == cfg.agent_size ** 2
    assert count(img, cfg.color_obstacle) == cfg.obstacle_size ** 2
    assert count(img, cfg.color_enemy) == cfg.enemy_size ** 2
    assert count(img, cfg.color_gem) == cfg.gem_size ** 2 == 4
    # positions exactes
    assert (img[10:14, 30:34] == cfg.color_agent).all()
    assert (img[56:58, 6:8] == cfg.color_gem).all()
    ex, ey = enemy_xy(s.eo, cfg)
    assert (img[ey[0]:ey[0] + 6, ex[0]:ex[0] + 6] == cfg.color_enemy).all()


def test_walls_and_pillars_are_drawn(cfg):
    img = render(make_state(cfg), cfg)[0]
    assert (img[0:2, :] == cfg.color_wall).all() and (img[:, 62:] == cfg.color_wall).all()
    assert (img[12:18, 12:18] == cfg.color_wall).all()
    assert (img[30, 20] == cfg.color_bg).all()


def test_small_object_stays_visible_over_enemy_and_obstacle(cfg):
    ex, ey = enemy_xy(np.array([0]), cfg)
    s = make_state(cfg, gx=int(ex[0]) + 2, gy=int(ey[0]) + 2, ox=int(ex[0]) - 1, oy=int(ey[0]) - 1)
    assert count(render(s, cfg)[0], cfg.color_gem) == 4


def test_image_does_not_reveal_the_next_branch(cfg):
    """L'image de s_t est calculée avant tout tirage : elle ne dépend que de s_t."""
    s = initial_state(cfg, 50, np.random.default_rng(1))
    s.eo[:] = 0
    before = render(s, cfg)
    sa, _ = advance(s, np.zeros(50, dtype=int), np.full(50, BRANCH_A), cfg)
    sb, _ = advance(s, np.zeros(50, dtype=int), np.full(50, BRANCH_B), cfg)
    assert np.array_equal(before, render(s, cfg))
    assert not np.array_equal(render(sa, cfg), render(sb, cfg))   # ... mais les futurs diffèrent


def test_counterfactual_images_differ_only_where_the_enemy_is(cfg):
    s = initial_state(cfg, 300, np.random.default_rng(2))
    s.eo[:] = 0
    act = np.random.default_rng(3).integers(0, 5, 300)
    img_a, img_b, sa, sb = render_counterfactual(s, act, cfg)
    diff = (img_a != img_b).any(axis=-1)                   # (B, 64, 64)
    ex_a, ey = enemy_xy(sa.eo, cfg)
    ex_b, _ = enemy_xy(sb.eo, cfg)
    E = cfg.enemy_size
    allowed = np.zeros_like(diff)
    for i in range(300):
        allowed[i, ey[i]:ey[i] + E, ex_a[i]:ex_a[i] + E] = True
        allowed[i, ey[i]:ey[i] + E, ex_b[i]:ex_b[i] + E] = True
    assert not (diff & ~allowed).any()
    assert diff.any(axis=(1, 2)).all()                     # A != B partout
    delta = branch_delta(img_a, img_b)
    assert delta.min() > 6.0 and delta.max() < 7.7         # δ ≈ 7,65 sans recouvrement, un peu moins sinon


def test_counterfactual_images_are_identical_away_from_the_hub(cfg):
    s = initial_state(cfg, 300, np.random.default_rng(4))
    s = s[~is_branch_state(s)]
    img_a, img_b, _, _ = render_counterfactual(s, np.full(s.n, NOOP), cfg)
    assert np.array_equal(img_a, img_b)
    assert (branch_delta(img_a, img_b) == 0).all()


def test_branch_delta_matches_closed_form(cfg):
    """Deux blocs 6x6 de couleur ennemie contre fond : δ² = 2·36·||c_ennemi - c_fond||²/255²."""
    s = make_state(cfg, ax=30, ay=10, ox=40, oy=45, eo=0, ev=-1, gx=6, gy=56)
    a, b, _, _ = render_counterfactual(s, [NOOP], cfg)
    diff = np.asarray(cfg.color_enemy, float) - np.asarray(cfg.color_bg, float)
    expected = np.sqrt(2 * cfg.enemy_size ** 2 * (diff ** 2).sum()) / 255.0
    assert np.isclose(branch_delta(a, b)[0], expected)
