"""Extension « décision », niveau 2 : planificateur, simulateurs factices, analyse (H0f, H5).

Les « modèles » sont le vrai jeu ou des variantes dont le comportement est connu ; aucun modèle appris.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import make_state
from flou.data.dataset import build_bifurcation_set
from flou.eval import decision as D
from flou.eval import decision_analysis as DA
from flou.eval import dryrun
from flou.eval import planning as P
from flou.eval import report as RP
from flou.eval.baselines import CopyLast
from flou.eval.predictors import Batch
from flou.game import DOWN, RIGHT, GameConfig, render
from flou.game.dynamics import advance
from flou.train import runs as R

ROOT = Path(__file__).resolve().parents[1]


def ctx_of(s, cfg, k=4):
    return np.repeat(render(s, cfg)[:, None], k, axis=1)


# ----------------------------------------------------------------------------- réglages
def test_plan_config_defaults_and_guards():
    pc = P.PlanConfig()
    assert (pc.candidates, pc.horizon, pc.samples, pc.lam, pc.reader) == (64, 8, 4, 0.5, "soft")
    assert pc.predictions_per_decision() == 64 * 4 * 8
    with pytest.raises(ValueError):
        P.PlanConfig(reader="flou")
    with pytest.raises(ValueError):
        P.PlanConfig(horizon=0)


# ----------------------------------------------------------------------------- simulateurs factices
def test_simulator_model_reproduces_the_game(cfg):
    one = cfg.with_p(1.0)                                            # branche A certaine : comparaison exacte
    s = make_state(one, eo=0, ev=1)
    sim = P.SimulatorModel(one)
    sim.begin_plan(s)
    rng = np.random.default_rng(0)
    cur = s
    for step, a in enumerate([1, 4, 4, 0, 2, 3, 3, 1]):             # 8 pas : passe par une seconde bifurcation
        y = sim.predict(Batch(np.arange(1), None, np.array([a]), step), rng)
        cur, _ = advance(cur, np.array([a]), np.array([1]), one)
        assert np.array_equal(y, render(cur, one))


def test_simulator_requires_begin_plan(cfg):
    with pytest.raises(RuntimeError, match="begin_plan"):
        P.SimulatorModel(cfg).predict(Batch(np.arange(1), None, np.array([0]), 0), np.random.default_rng(0))
    with pytest.raises(ValueError, match="états exacts"):
        P.Planner(P.SimulatorModel(cfg), cfg, P.PlanConfig(candidates=2, horizon=1, samples=1)).act(
            ctx_of(make_state(cfg), cfg), np.random.default_rng(0))


def test_mean_simulator_is_the_exact_mixture_and_merges_at_the_hub(cfg):
    p = cfg.p
    s = make_state(cfg, eo=0, ev=1)
    sim = P.SimulatorModel(cfg, "mean")
    sim.begin_plan(s)
    rng = np.random.default_rng(0)
    a_state, b_state = s, s
    acts = [0, 1, 0, 2, 0, 0, 0]
    for step, a in enumerate(acts):
        y = sim.predict(Batch(np.arange(1), None, np.array([a]), step), rng)
        if step < 6:                                                # avant le retour au centre : deux futurs
            a_state, _ = advance(a_state, np.array([a]), np.array([1]), cfg)
            b_state, _ = advance(b_state, np.array([a]), np.array([0]), cfg)
            assert (a_state.eo[0] == b_state.eo[0]) == (step == 5)  # confondus au 6e pas, au centre du rail
        else:                                                       # nouvelle bifurcation : de nouveau p / 1-p
            a_state, _ = advance(a_state, np.array([a]), np.array([1]), cfg)
            b_state, _ = advance(b_state, np.array([a]), np.array([0]), cfg)
        want = p * render(a_state, cfg).astype(np.float64) + (1 - p) * render(b_state, cfg).astype(np.float64)
        assert np.abs(y.astype(np.float64) - want).max() <= 0.5      # au pixel près (quantification 8 bits)


def test_fade_erases_the_enemy(cfg):
    s = make_state(cfg, eo=0, ev=1)
    rd = D.ContactReader(cfg)
    imgs = {}
    for fade in (1.0, 0.0):
        sim = P.SimulatorModel(cfg, "mean", fade=fade)
        sim.begin_plan(s)
        imgs[fade] = sim.predict(Batch(np.arange(1), None, np.array([0]), 0), np.random.default_rng(0))
    y0 = cfg.hub[1]
    red = lambda im: (im[:, y0:y0 + cfg.enemy_size, :, 0].astype(int) - im[:, y0:y0 + cfg.enemy_size, :, 1]).max()   # sur le rail
    assert red(imgs[1.0]) > 100 and red(imgs[0.0]) < 20              # ennemi visible / effacé
    assert rd.agent(imgs[0.0])[2].all()                              # l'agent, lui, reste


# ----------------------------------------------------------------------------- le planificateur décide correctement
def escape_state(cfg):
    """Agent juste à droite du centre du rail, en bas : s'il ne descend pas, l'ennemi le touche dans la branche A."""
    hx, hy = cfg.hub
    return make_state(cfg, ax=hx + cfg.enemy_size + 1, ay=hy + cfg.enemy_size - 2, eo=0, ev=1, gx=6, gy=6)


def test_planner_escapes_a_certain_contact(cfg):
    one = cfg.with_p(1.0)
    s = escape_state(one)
    pc = P.PlanConfig(candidates=64, horizon=1, samples=1, lam=0.0)
    for reader in ("soft", "hard"):
        pl = P.Planner(P.SimulatorModel(one), one, P.PlanConfig(candidates=64, horizon=1, samples=1, lam=0.0, reader=reader))
        assert pl.act(ctx_of(s, one), np.random.default_rng(0), s)[0] == DOWN
    assert pc.predictions_per_decision() == 64


def test_soft_reader_sees_a_30_percent_risk_in_the_mean_image_and_the_threshold_does_not(cfg):
    s = escape_state(cfg)                                            # p = 0,3 : risque 0,3 si l'agent ne descend pas
    rng = np.random.default_rng(1)
    soft = P.Planner(P.SimulatorModel(cfg, "mean"), cfg, P.PlanConfig(candidates=64, horizon=1, samples=1, lam=0.0))
    assert soft.act(ctx_of(s, cfg), rng, s)[0] == DOWN
    # lecteur à seuil : 0,3 < 0,5 est lu « pas de contact » ; toutes les actions se valent, la première suite gagne
    hard = P.Planner(P.SimulatorModel(cfg, "mean"), cfg,
                     P.PlanConfig(candidates=64, horizon=1, samples=1, lam=0.0, reader="hard"))
    seqs = np.random.default_rng(5).integers(0, 5, size=(1, 64, 1))
    assert hard.act(ctx_of(s, cfg), np.random.default_rng(5), s)[0] == seqs[0, 0, 0]


def test_planner_picks_up_a_reachable_gem_and_walks_towards_a_far_one(cfg):
    near = make_state(cfg, ax=2, ay=6, gx=6, gy=6, eo=2, ev=1)
    pl = P.Planner(P.SimulatorModel(cfg), cfg, P.PlanConfig(candidates=64, horizon=1, samples=1, lam=0.0))
    assert pl.act(ctx_of(near, cfg), np.random.default_rng(0), near)[0] == RIGHT
    far = make_state(cfg, ax=20, ay=30, gx=56, gy=31, eo=-3, ev=1, ox=40, oy=6)
    pl = P.Planner(P.SimulatorModel(cfg), cfg, P.PlanConfig(candidates=64, horizon=1, samples=1, lam=0.5))
    assert pl.act(ctx_of(far, cfg), np.random.default_rng(0), far)[0] == RIGHT


def test_planner_chunks_give_one_action_per_game(cfg):
    from flou.game.dynamics import initial_state
    s = initial_state(cfg, 7, np.random.default_rng(0))
    pl = P.Planner(P.SimulatorModel(cfg), cfg, P.PlanConfig(candidates=8, horizon=2, samples=2, max_rows=40))
    a = pl.act(ctx_of(s, cfg), np.random.default_rng(0), s)
    assert a.shape == (7,) and ((a >= 0) & (a < 5)).all()
    assert pl.n_predictions == 7 * 8 * 2 * 2 and pl.n_lost == 0


# ----------------------------------------------------------------------------- parties
SMALL = P.PlanConfig(candidates=16, horizon=4, samples=2)


def test_play_is_reproducible_and_counts_predictions(cfg):
    a = P.play(P.SimulatorModel(cfg), cfg, SMALL, n_games=3, seed=4, steps=6)
    b = P.play(P.SimulatorModel(cfg), cfg, SMALL, n_games=3, seed=4, steps=6)
    assert a["per_game"] == b["per_game"]
    assert a["predictions"] == 3 * 6 * SMALL.predictions_per_decision() and a["agent_lost"] == 0.0
    assert a["score"] == pytest.approx(a["pickups"] - a["contacts"])


def test_play_works_with_an_image_only_predictor(cfg):
    """Chemin réel (modèle appris) : le prédicteur ne reçoit que le contexte d'images, pas l'état."""
    seen = []

    class Spy(CopyLast):
        def predict(self, batch, rng):
            seen.append((batch.ctx.shape, batch.ctx.dtype, batch.step))
            return super().predict(batch, rng)

    pc = P.PlanConfig(candidates=4, horizon=3, samples=1)
    res = P.play(Spy(), cfg, pc, n_games=2, seed=0, steps=2)
    assert res["predictions"] == 2 * 2 * 4 * 3 and res["model"] == "copie-derniere-image"
    assert seen[0] == ((8, 4, 64, 64, 3), np.uint8, 0) and [s[2] for s in seen[:3]] == [0, 1, 2]


def test_games_do_not_depend_on_the_model(cfg):
    """Comparaison appariée : mêmes états initiaux et mêmes tirages de branche quel que soit le planificateur."""
    from flou.game import VecMiniGame
    e1, e2 = VecMiniGame(cfg, 4, seed=9), VecMiniGame(cfg, 4, seed=9)
    e1.reset(), e2.reset()
    assert e1.state == e2.state
    rng = np.random.default_rng(0)
    b1 = [e1.step(rng.integers(0, 5, 4))[4]["branch"] for _ in range(12)]
    b2 = [e2.step(np.zeros(4, dtype=int))[4]["branch"] for _ in range(12)]
    assert all(np.array_equal(x, y) for x, y in zip(b1, b2))


def test_h0f_control_structure_and_scaled_thresholds(cfg):
    rep = P.check_planner(cfg, SMALL, n_games=3, seed=0, steps=10)
    assert set(rep) >= {"planner", "random", "expert", "gain_over_random", "pass"}
    assert rep["max_contacts"] == pytest.approx(0.05) and rep["min_gain"] == pytest.approx(0.3)
    assert rep["planner"]["n_games"] == rep["random"]["n_games"] == 3


# ----------------------------------------------------------------------------- analyse : règles de décision
def _drec(family, seed, soft, hard, variant=None):
    return {"run_id": f"core-{family}-s{seed}", "group": "core", "family": family, "p": 0.3, "seed": seed, "K": None,
            "variant": DA.MAIN_VARIANT[family] if variant is None else variant, "m": 8, "ks": [1],
            "soft_k1": soft, "hard_k1": hard, "other_soft_k1": 0.0, "n_amb_k1": 100, "found_k1": 1.0}


def _prec(family, seed, contacts, score=0.0):
    return {"run_id": f"core-{family}-s{seed}", "group": "core", "family": family, "p": 0.3, "seed": seed, "K": None,
            "variant": DA.MAIN_VARIANT[family], "score": score, "contacts": contacts, "pickups": 1.0, "n_games": 50,
            "reader": "soft", "samples": 4, "agent_lost": 0.0, "seconds": 1.0}


def test_verdicts_without_any_measurement_are_not_evaluable():
    v = DA.verdicts([], [], None)
    assert set(v) == set(DA.HYPOTHESES)
    assert all(x["status"] == "non évaluable" for x in v.values())


@pytest.mark.parametrize("reg,dif,status", [
    ([0.30, 0.31, 0.29], [0.14, 0.15, 0.16], "confirmée"),
    ([0.20, 0.21, 0.19], [0.14, 0.15, 0.16], "réfutée"),                # écart 0,06 < 0,10
    ([0.10, 0.30, 0.50], [0.10, 0.20, 0.25], "non concluante"),         # écart 0,117 mais trop dispersé
])
def test_h4a_rule(reg, dif, status):
    d = [_drec("regression", i, 0.0, x) for i, x in enumerate(reg)] + \
        [_drec("diffusion", i, x, x) for i, x in enumerate(dif)]
    assert DA.verdicts(d, [], None)["H4a"]["status"] == status


@pytest.mark.parametrize("reg,dif,status", [(0.00, 0.14, "confirmée"), (0.15, 0.14, "confirmée"),
                                             (0.18, 0.14, "non concluante"), (0.25, 0.14, "réfutée")])
def test_h4b_rule(reg, dif, status):
    d = [_drec("regression", 0, reg, 0.3), _drec("diffusion", 0, dif, dif)]
    assert DA.verdicts(d, [], None)["H4b"]["status"] == status


def test_h4_ignores_the_wrong_diffusion_variant():
    d = [_drec("regression", i, 0.0, 0.3) for i in range(3)] + \
        [_drec("diffusion", i, 0.0, 0.3, variant="n=1") for i in range(3)]
    v = DA.verdicts(d, [], None)
    assert v["H4a"]["status"] == "non évaluable" and v["H4b"]["status"] == "non évaluable"


@pytest.mark.parametrize("reg,dif,status", [
    ([2.0, 2.2, 1.8], [0.2, 0.3, 0.1], "confirmée"),
    ([0.03, 0.05, 0.04], [0.15, 0.2, 0.1], "réfutée"),                  # la régression fait MIEUX : réfutée
    ([0.2, 1.5, 0.6], [0.1, 0.3, 0.2], "non concluante"),
])
def test_h5a_rule(reg, dif, status):
    pr = [_prec("regression", i, x) for i, x in enumerate(reg)] + [_prec("diffusion", i, x) for i, x in enumerate(dif)]
    assert DA.verdicts([], pr, None)["H5a"]["status"] == status


def test_h5b_needs_matching_eval_records():
    pr = [_prec(f, i, 0.1 * i, score=float(i + (f == "diffusion"))) for f in ("regression", "diffusion") for i in range(3)]
    ev = [{"run_id": r["run_id"], "variant": r["variant"], "sharp": r["score"] / 10, "mse_per_pixel": 1 - r["score"] / 10,
           "track_last": 0.5} for r in pr]
    v = DA.verdicts([], pr, None, ev)["H5b"]
    assert v["status"].startswith("exploratoire") and v["n_points"] == 6
    assert v["spearman_score_nettes"] == pytest.approx(1.0) and v["spearman_score_mse_pixel"] == pytest.approx(-1.0)
    assert np.isnan(v["spearman_score_suivi_16"])                    # série constante : indéfinie
    assert DA.verdicts([], pr, None, ev[:2])["H5b"]["status"] == "non évaluable"


# ----------------------------------------------------------------------------- répétition générale et scripts
@pytest.fixture(scope="module")
def simdec(tmp_path_factory):
    root = tmp_path_factory.mktemp("simdec")
    cfg = GameConfig().with_p(0.3)
    dset = build_bifurcation_set(cfg, n=400, seed=1000)
    sub = dset.subset(D.select_states(dset, (1, 3), seed=0))
    plan = [r for r in R.build_plan(steps=100) if r.group == "core" and r.seed in (0, 1)]
    for r in plan:
        dryrun.simulate_decision(r, sub, root / "runs", ks=(1, 3), m=4)
        if r.family != "tokens":
            dryrun.simulate_planning(r, cfg, root / "runs", P.PlanConfig(candidates=16, horizon=6, samples=2),
                                     n_games=6, game_steps=60)
    return root


def test_simulated_decision_files_feed_the_analysis(simdec):
    assert RP.is_simulated(simdec / "runs")
    drecs, precs = DA.load_decision(simdec / "runs"), DA.load_planning(simdec / "runs")
    assert len(drecs) == 2 * 4 and len(precs) == 4                   # diffusion : n = 1 et n = 3
    js = json.loads(next((simdec / "runs").glob("*regression*/planning.json")).read_text(encoding="utf-8"))
    assert js["SIMULATION"] is True and js["variants"][0]["play"]["config"]["samples"] == 1
    v = DA.verdicts(drecs, precs, None)
    assert v["H4a"]["status"] == "confirmée" and v["H4b"]["status"] == "confirmée"
    assert v["H5a"]["ecart_moyen"] > 0.5                              # l'effet planté (ennemi effacé) est retrouvé
    assert v["H0e"]["status"] == "non évaluable"
    rt = DA.risk_table(drecs)
    assert [r["modèle"] for r in rt] == ["A. Régression", "B. Diffusion (EDM) n=1", "B. Diffusion (EDM) n=3", "C. Jetons K=256"]
    assert rt[0]["doux_k1"] < 0.01 and rt[0]["seuil_k1"] == pytest.approx(0.3, abs=1e-6)
    md = "\n".join(DA.report_section(drecs, precs, None))
    assert "Niveau 1" in md and "Niveau 2" in md and "docs/EXTENSION_DECISION.md" in md


def test_report_gets_the_decision_section_only_when_asked(simdec):
    drecs, precs = DA.load_decision(simdec / "runs"), DA.load_planning(simdec / "runs")
    ver = DA.verdicts(drecs, precs, None)
    md = RP.render_report([], ver, simulated=True, decision={"risk": drecs, "planning": precs, "instrument": None})
    assert "Extension « décision »" in md and "| H5a |" in md and "Lambert" in md
    assert "Extension « décision »" not in RP.render_report([], {}, simulated=True)


def run_script(name, *args, cwd):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, timeout=300)


def test_validate_decision_script(tmp_path):
    r = run_script("validate_decision.py", "--no-planner", "--states", "400", "--m", "4", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "data" / "p0.30" / "test_decision.npz").exists()
    rep = json.loads((tmp_path / "reports" / "decision_instrument.json").read_text(encoding="utf-8"))
    assert rep["h0e"]["pass"] and rep["h0f"] is None and rep["n_states"] == 400
    assert "H0e" in r.stdout and "non lancé" in r.stdout
    r = run_script("validate_decision.py", "--games", "2", "--game-steps", "4", "--candidates", "8", "--samples", "1",
                   "--horizon", "3", cwd=tmp_path)                    # H0f minuscule : réutilise le jeu de test créé
    assert "créé" not in r.stdout and "H0f (planificateur avec le vrai jeu)" in r.stdout
    rep = json.loads((tmp_path / "reports" / "decision_instrument.json").read_text(encoding="utf-8"))
    assert rep["h0f"]["planner"]["config"]["candidates"] == 8


def test_analyze_script_reports_decision_hypotheses(simdec, tmp_path):
    plan = [r for r in R.build_plan(steps=100) if r.group == "core" and r.seed in (0, 1)]
    dryrun.simulate_plan(plan, tmp_path / "data", simdec / "runs", rho_max=0.165, n_states=120, seed=2)
    r = run_script("analyze.py", "--runs", simdec / "runs", "--no-figures", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "reports" / "dryrun"
    ver = json.loads((out / "verdicts.json").read_text(encoding="utf-8"))
    assert {"H0e", "H0f", "H4a", "H4b", "H4c", "H5a", "H5b"} <= set(ver)
    assert ver["H4a"]["status"] == "confirmée" and ver["H5b"]["status"].startswith("exploratoire")
    assert (out / "table_risque.csv").exists() and (out / "table_planification.csv").exists()
    assert "Extension « décision »" in (out / "report.md").read_text(encoding="utf-8")


def test_evaluate_decision_script_needs_pytorch_and_says_so(tmp_path):
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("PyTorch est installé")
    r = run_script("evaluate_decision.py", "--all", cwd=tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "PyTorch est requis" in out and "Traceback" not in out
