import json

import pytest

from flou.game import MIN_BRANCH_DIST, GameConfig


def test_default_config_is_valid(cfg):
    assert cfg.size == 64
    assert cfg.branch_distance >= MIN_BRANCH_DIST
    assert cfg.hub == (29, 29)


def test_with_p_changes_only_p(cfg):
    c2 = cfg.with_p(0.5)
    assert c2.p == 0.5 and cfg.p == 0.3
    assert c2.to_dict() | {"p": 0.3} == cfg.to_dict()


def test_json_roundtrip(cfg, tmp_path):
    path = tmp_path / "cfg.json"
    cfg.to_json(path)
    assert GameConfig.from_json(path) == cfg
    assert GameConfig.from_dict(json.loads(cfg.to_json())) == cfg


def test_config_is_hashable(cfg):
    assert hash(cfg) == hash(GameConfig())


def test_unknown_parameter_rejected(cfg):
    with pytest.raises(ValueError, match="inconnus"):
        GameConfig.from_dict({**cfg.to_dict(), "typo": 1})


def test_branch_distance_below_minimum_rejected():
    with pytest.raises(ValueError, match="distance entre branches"):
        GameConfig(enemy_step=3)


def test_pillar_on_rail_rejected():
    with pytest.raises(ValueError, match="rail"):
        GameConfig(pillars=((20, 28, 6, 6),))


def test_p_out_of_range_rejected():
    with pytest.raises(ValueError):
        GameConfig(p=1.5)
