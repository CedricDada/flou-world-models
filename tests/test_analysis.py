"""Répétition générale : analyse de bout en bout sur des runs SIMULÉS par des prédicteurs factices.

Rien ici ne mesure un effet : on vérifie que l'analyse reconnaît chaque effet qu'on y a mis, et ne
conclut pas quand les données ne le permettent pas.
"""
import json

import numpy as np
import pytest

from flou.eval import analysis as A
from flou.eval import dryrun, instrument
from flou.train import runs as R


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("dry")
    plan = R.build_plan(steps=100)
    dryrun.simulate_plan(plan, root / "data", root / "runs", rho_max=0.16, n_states=300, seed=3)
    files = sorted((root / "data").glob("p*/test_bifurcation.npz"))
    inst = instrument.validate_instrument(files, noise_sigma=0.01)
    recs = A.load_records(root / "runs")
    return {"root": root, "plan": plan, "recs": recs, "inst": inst}


def test_plan_has_25_runs_in_the_course_proportions():
    plan = R.build_plan()
    assert R.summarize_plan(plan) == {"core": 9, "control": 3, "k_axis": 6, "grid": 1, "p_axis": 6, "total": 25}
    assert len({r.run_id for r in plan}) == 25
    order = [r.group for r in plan]
    # ordre de priorité : cœur et contrôles avant l'axe K, avant la grille, avant l'axe p (sacrifié en premier)
    assert order.index("k_axis") > max(i for i, g in enumerate(order) if g in ("core", "control"))
    assert order.index("grid") > max(i for i, g in enumerate(order) if g == "k_axis")
    assert order.index("p_axis") > order.index("grid")
    assert {r.seed for r in plan if r.group == "core"} == {0, 1, 2}
    assert sorted({r.K for r in plan if r.family == "tokens"}) == [64, 256, 1024]
    assert [r for r in plan if r.group == "grid"][0].grid == 16


def test_runspec_validation_and_roundtrip(tmp_path):
    with pytest.raises(ValueError):
        R.RunSpec("tokens", "core", 0.3, 0)                    # K manquant
    with pytest.raises(ValueError):
        R.RunSpec("regression", "core", 0.3, 0, K=64)          # K interdit hors jetons
    with pytest.raises(ValueError):
        R.RunSpec("regression", "bogus", 0.3, 0)
    s = R.RunSpec("tokens", "k_axis", 0.3, 2, K=64, steps=1000)
    s.save(tmp_path)
    assert R.RunSpec.load(tmp_path) == s
    assert s.stage_steps() == {"vq": 400, "gpt": 600} and s.images_seen == 1000 * 64
    assert R.RunSpec("regression", "core", 0.3, 0).stage_steps() == {"main": 20000}


def test_matched_budget_across_families():
    plan = R.build_plan(steps=5000, batch_size=64)
    core = [r for r in plan if r.group == "core"]
    assert len({r.images_seen for r in core}) == 1                 # mêmes images vues
    assert len({r.batch_size for r in core}) == 1
    assert all(sum(r.stage_steps().values()) == r.steps for r in plan)   # C : VQ-VAE + Transformer = même total


def test_estimated_hours_scale_with_steps():
    plan = R.build_plan(steps=1000)
    sps = {"regression": 0.3, "oracle": 0.3, "diffusion": 0.4, "tokens-vq": 0.2, "tokens-gpt": 0.3}
    h1 = R.estimate_hours(plan, sps)
    h2 = R.estimate_hours(R.with_steps(plan, 2000), sps)
    assert h2 == pytest.approx(2 * h1, rel=0.01)


def test_records_cover_all_runs(world):
    ids = {r["run_id"] for r in world["recs"]}
    assert ids == {r.run_id for r in world["plan"]}
    dif = A.select(world["recs"], family="diffusion", group="core", seed=0)
    assert sorted(r["n_steps"] for r in dif) == [1, 3, 5, 20]


def test_verdicts_recognise_the_planted_effects(world):
    v = A.verdicts(world["recs"], world["inst"], min_n_h2b=100)
    assert v["H0a"]["status"] == "réussi"
    assert v["H0b"]["status"] == "réussi" and not v["H0b"]["fuite_detectee"]
    assert v["H0c"]["status"] == "réussi" and v["H0d"]["status"] == "réussi"
    assert v["H0d-diffusion(n=1)"]["status"] == "réussi"
    assert v["H1"]["status"] == "confirmée"
    assert v["H2a"]["status"] == "confirmée"
    # le biais d'Euler du cas jouet (σmax ≈ 5,3 δ) est mis dans les prédicteurs factices : p̂(n=3) ≈ 0,19
    # au lieu de 0,3, donc une erreur d'environ 0,11 > 0,05 : H2b réfutée
    assert v["H2b"]["status"] == "réfutée" and all(0.05 < e < 0.2 for e in v["H2b"]["erreur_absolue"])
    assert v["H3a"]["status"] == "confirmée" and v["H3a"]["ecart_moyen"] < -0.2
    assert v["H3b"]["status"].startswith("exploratoire") and v["H3b"]["n_points"] == 9


def test_h2b_is_inconclusive_without_enough_sharp_images(world):
    v = A.verdicts(world["recs"], world["inst"])           # min_n_h2b = 1000, mais 300 états seulement
    assert v["H2b"]["status"] == "non concluante"


def test_verdicts_do_not_conclude_with_missing_runs(world):
    only_reg = [r for r in world["recs"] if r["family"] == "regression" and r["group"] == "core"]
    v = A.verdicts(only_reg, None)
    assert v["H0a"]["status"] == "non évaluable" and v["H1"]["status"] == "non évaluable"
    assert v["H2a"]["status"] == "non évaluable" and v["H3a"]["status"] == "non évaluable"
    assert v["H0b"]["status"] == "réussi"                  # ce qui est évaluable l'est


def test_h1_is_refuted_when_the_regressor_does_not_blur(world):
    recs = json.loads(json.dumps(world["recs"], default=float))
    for r in recs:
        if r["family"] == "regression" and r["group"] == "core":
            r["blur"] = 0.1
    assert A.verdicts(recs, world["inst"])["H1"]["status"] == "réfutée"


def test_h3a_is_inconclusive_when_the_gap_is_inside_the_noise(world):
    recs = json.loads(json.dumps(world["recs"], default=float))
    rng = np.random.default_rng(0)
    for r in recs:
        if r["family"] == "tokens" and r["K"] in (64, 1024):
            r["recall"] = 0.7 + rng.normal(0, 0.2) * (1 if r["K"] == 1024 else 1) - (0.05 if r["K"] == 64 else 0.0)
    v = A.verdicts(recs, world["inst"])["H3a"]
    assert v["ecart_moyen"] < 0 or v["status"] == "réfutée"
    assert v["status"] in ("non concluante", "réfutée", "confirmée")
    # un écart de 0,05 sous un écart-type de 0,2 ne peut pas être « concluant » à 3 graines
    for r in recs:
        if r["family"] == "tokens" and r["K"] == 64:
            r["recall"] = 0.69
        if r["family"] == "tokens" and r["K"] == 1024:
            r["recall"] = 0.7 + (0.3 if r["seed"] == 0 else -0.3 if r["seed"] == 1 else 0.0)
    assert A.verdicts(recs, world["inst"])["H3a"]["status"] == "non concluante"


def test_tables_and_csv(world, tmp_path):
    tab = A.model_error_table(world["recs"])
    names = {r["modèle"] for r in tab}
    assert any(n.startswith("A.") for n in names) and any("n=3" in n for n in names) and any("K=64" in n for n in names)
    reg = next(r for r in tab if r["modèle"].startswith("A."))
    assert reg["flou"] == pytest.approx(1.0) and reg["mse_sur_plancher"] == pytest.approx(1.0, abs=0.01)
    d3 = next(r for r in tab if "n=3" in r["modèle"])
    assert d3["flou"] == 0.0 and d3["effondrement_de_mode"] == 0.0           # net, biaisé (p̂ ≈ 0,19) mais pas effondré
    collapsed = json.loads(json.dumps(world["recs"], default=float))
    for r in collapsed:
        if r["family"] == "diffusion" and r["n_steps"] == 3 and r["group"] == "core":
            r["p_hat"] = 0.0                                                    # une seule branche produite
    assert next(r for r in A.model_error_table(collapsed) if "n=3" in r["modèle"])["effondrement_de_mode"] == 1.0
    k64 = next(r for r in tab if "K=64" in r["modèle"])
    k1024 = next(r for r in tab if "K=1024" in r["modèle"])
    assert k64["perte_petit_objet"] > k1024["perte_petit_objet"] + 0.2
    path = A.write_csv(tab, tmp_path / "t.csv")
    assert path.read_text(encoding="utf-8").splitlines()[0].startswith("modèle,graines")
    assert "| modèle |" in A.markdown_table(tab)
    cal = A.calibration_by_n(world["recs"])
    assert [c["n"] for c in cal] == [1, 3, 5, 20] and 0.1 < cal[1]["p_hat_moyen"] < 0.25
    comp = A.compute_report(world["recs"])
    assert len(comp) == 25 and comp[0]["heures_gpu"] == 1.5


def test_figures_are_written(world, tmp_path):
    pytest.importorskip("matplotlib")
    from flou.eval import analysis_figs as F
    made = F.all_figures(world["recs"], tmp_path)
    assert {m.name for m in made} == {"fig_flou.png", "fig_flou_selon_p.png", "fig_calibration.png",
                                      "fig_rappel.png", "fig_rollout.png", "fig_mse_plancher.png"}
    assert all(m.stat().st_size > 5_000 for m in made)


def test_dryrun_outputs_are_labelled_as_simulation(world):
    d = next((world["root"] / "runs").iterdir())
    ev = json.loads((d / "eval.json").read_text(encoding="utf-8"))
    assert ev["SIMULATION"] is True and ev["probe"] == "SIMULATION"
    assert json.loads((d / "compute.json").read_text(encoding="utf-8"))["gpu"] == "SIMULATION"
