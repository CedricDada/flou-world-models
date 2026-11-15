"""Chaîne d'outils autour des runs : tableaux sauvegardés, relecture à partir des runs, rapport, scripts en ligne
de commande. Les runs sont SIMULÉS (flou.eval.dryrun) : on teste la plomberie, pas des résultats.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from flou.data.dataset import build_bifurcation_set
from flou.eval import analysis as A
from flou.eval import arrays, dryrun, human, instrument
from flou.eval import classifier as C
from flou.eval import report as RP
from flou.game import GameConfig
from flou.train import runs as R

ROOT = Path(__file__).resolve().parents[1]


def run_script(name, *args, cwd):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, timeout=240)


# ----------------------------------------------------------------------------- blocs du classifieur
@pytest.fixture(scope="module")
def pairs():
    bs = build_bifurcation_set(GameConfig().with_p(0.3), n=40, seed=0)
    a, b, _ = bs.counterfactuals()
    nxt, _ = bs.next_states()
    return a, b, np.stack([nxt.gx, nxt.gy], axis=1)


def test_h0a_is_identical_to_the_unchunked_path_for_small_sets(pairs):
    """Jusqu'à CHUNK paires, découper en blocs ne change pas un seul tirage aléatoire."""
    a, b, gem = pairs
    new = C.h0a_check(a, b, gem, rho_max=0.17, noise_sigma=0.01, seed=5)
    syn = C.make_synthetic(a, b, np.random.default_rng(5), noise_sigma=0.01, gem_xy=gem)
    old = C.evaluate_synthetic(syn, a, b, 0.17)
    for k, v in old.items():
        assert new[k] == pytest.approx(v, nan_ok=True), k


def test_chunking_keeps_the_statistics(pairs):
    a, b, gem = pairs
    big = C.synthetic_stats(a, b, np.random.default_rng(1), 0.01, gem_xy=gem, chunk=1000)
    small = C.synthetic_stats(a, b, np.random.default_rng(1), 0.01, gem_xy=gem, chunk=7)
    assert np.bincount(big["kind"]).tolist() == np.bincount(small["kind"]).tolist() == [40, 40, 40, 40]
    for k in (0, 1):                                              # images pures : ρ ≈ √D σ / δ, ne dépend pas du découpage
        r1, r2 = big["rho"][big["kind"] == k].mean(), small["rho"][small["kind"] == k].mean()
        assert r1 == pytest.approx(r2, rel=0.02)
    assert np.nanmean(np.abs(small["alpha_hat"][small["kind"] == 0] - 1.0)) < 0.01


def test_synthetic_stats_without_mixtures(pairs):
    a, b, _ = pairs
    st = C.synthetic_stats(a, b, np.random.default_rng(0), 0.01, with_mix=False, chunk=16)
    assert set(np.unique(st["kind"]).tolist()) == {0, 1} and len(st["rho"]) == 80


def test_small_sample_warning():
    assert instrument.small_sample_warning(3000) is None
    assert instrument.small_sample_warning(600) is None
    w = instrument.small_sample_warning(120)
    assert "120 paires" in w and "taille" in w


# ----------------------------------------------------------------------------- tableaux sauvegardés
def _res(label, ordered=True, n_samples=1, hw=8):
    n_states = len(label) // n_samples
    state = np.tile(np.arange(n_states), n_samples)
    if not ordered:
        state = state[::-1].copy()
    imgs = np.random.default_rng(0).random((len(label), hw, hw, 3)).astype(np.float32)
    return {"per_sample": {"alpha": np.zeros(len(label)), "rho": np.zeros(len(label)), "label": label,
                           "recall": np.ones(len(label), bool), "state": state},
            "images": imgs, "n_states": n_states}


def test_pack_arrays_caps_each_class_and_keeps_uint8():
    label = np.repeat([0, 1, 2, 3], [120, 60, 10, 0]).astype(np.int8)
    out = arrays.pack_arrays(_res(label), "n=3", per_class=25, n_ab=40)
    idx = out["n=3__img_idx"]
    assert np.bincount(label[idx], minlength=4).tolist() == [25, 25, 10, 0]      # plafonné, classe rare entière
    assert (np.diff(idx) > 0).all() and out["n=3__img"].dtype == np.uint8 and len(out["n=3__img"]) == len(idx)
    assert out["n=3__ab_img"].shape == (40, 8, 8, 3)
    assert set(out) >= {f"n=3__{k}" for k in ("alpha", "rho", "label", "recall", "state")}


def test_pack_arrays_ab_images_come_from_the_first_draw_only():
    label = np.repeat([0, 1], 50).astype(np.int8)
    out = arrays.pack_arrays(_res(label, n_samples=2), "main", n_ab=30)
    assert out["main__ab_img"].shape[0] == 30                                     # états 0..29, tirage 0
    assert "main__ab_img" not in arrays.pack_arrays(_res(label, ordered=False), "main")   # pas rangé par état


def test_pack_arrays_without_images_keeps_the_per_sample_arrays_only():
    res = _res(np.zeros(10, np.int8))
    res.pop("images")
    assert set(arrays.pack_arrays(res, "main")) == {f"main__{k}" for k in ("alpha", "rho", "label", "recall", "state")}


def test_variant_key():
    assert arrays.variant_key("") == "main" and arrays.variant_key("n=3") == "n=3"


# ----------------------------------------------------------------------------- runs simulés
@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    plan = [r for r in R.build_plan(steps=100) if r.group == "core" and r.seed == 0]
    dryrun.simulate_plan(plan, root / "data", root / "runs", rho_max=0.165, n_states=200, seed=2)
    return {"root": root, "plan": plan, "ids": [r.run_id for r in plan]}


def test_simulated_runs_save_arrays_with_images(sim):
    z = np.load(sim["root"] / "runs" / sim["ids"][1] / "eval_arrays.npz")
    assert {"n=1__img", "n=3__img", "n=5__img", "n=20__img", "n=3__ab_img"} <= set(z.files)
    assert z["n=3__img"].dtype == np.uint8 and z["n=3__img"].shape[1:] == (64, 64, 3)


def test_parse_selection():
    assert human.parse_selection("core-diffusion-p0.30-s0:n=3") == ("core-diffusion-p0.30-s0", "n=3")
    assert human.parse_selection("core-regression-p0.30-s0") == ("core-regression-p0.30-s0", "")


def test_load_pool_pools_variants_and_keeps_the_full_population(sim):
    ids = sim["ids"]
    sel = [(ids[0], ""), (ids[1], "n=3"), (ids[2], "tau=1")]
    pool = human.load_pool(sim["root"] / "runs", sim["root"] / "data", sel)
    n = len(pool["labels"])
    assert pool["imgs"].shape == (n, 64, 64, 3) and pool["A"].shape == pool["B"].shape == pool["imgs"].shape
    assert len(pool["source"]) == n and set(pool["source"]) == {f"{ids[0]}:main", f"{ids[1]}:n=3", f"{ids[2]}:tau=1"}
    assert len(pool["population"]) == 3 * 200 and n < len(pool["population"])    # réservoir plafonné, population entière
    # la référence A ou B est bien celle de l'état prédit : une image pure (classe nette) lui ressemble
    for cls, ref in ((0, "A"), (1, "B")):
        m = pool["labels"] == cls
        if m.any():
            d = np.abs(pool["imgs"][m].astype(float) - pool[ref][m].astype(float)).mean(axis=(1, 2, 3)) / 255.0
            assert d.max() < 0.05, f"nette-{ref} éloignée de sa référence"


def test_load_pool_errors(sim):
    with pytest.raises(KeyError, match="variantes présentes"):
        human.load_pool(sim["root"] / "runs", sim["root"] / "data", [(sim["ids"][0], "n=99")])
    with pytest.raises(FileNotFoundError):
        human.load_pool(sim["root"] / "runs", sim["root"] / "data", [("n-existe-pas", "")])


def test_load_ab_pair_uses_same_contexts_and_checks_p(sim, tmp_path):
    ids = sim["ids"]
    ab = human.load_ab_pair(sim["root"] / "runs", sim["root"] / "data", (ids[1], "n=3"), (ids[0], ""))
    assert ab["pred_a"].shape == ab["pred_b"].shape == ab["ctx_last"].shape == (40, 64, 64, 3)
    assert ab["name_a"] == f"{ids[1]}:n=3" and ab["name_b"] == ids[0]
    # même p exigé : un faux run à p = 0,5 est refusé
    import shutil
    other = sim["root"] / "runs" / "faux-p050"
    shutil.copytree(sim["root"] / "runs" / ids[0], other)
    spec = json.loads((other / "spec.json").read_text(encoding="utf-8"))
    spec["p"] = 0.5
    (other / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    cfg5 = GameConfig().with_p(0.5)
    bs = build_bifurcation_set(cfg5, n=50, seed=3)
    bs.save(sim["root"] / "data" / "p0.50" / "test_bifurcation.npz", {"config": cfg5.to_dict()})
    with pytest.raises(ValueError, match="même p"):
        human.load_ab_pair(sim["root"] / "runs", sim["root"] / "data", (ids[1], "n=3"), ("faux-p050", ""))
    shutil.rmtree(other)                                            # le fixture est partagé : on ne laisse rien derrière
    shutil.rmtree(sim["root"] / "data" / "p0.50")


# ----------------------------------------------------------------------------- rapport
def test_report_flags_simulation_and_escapes_pipes(sim):
    recs = A.load_records(sim["root"] / "runs")
    ver = A.verdicts(recs, None)
    md = RP.render_report(recs, ver, R.build_plan(), sim["root"] / "runs", simulated=True)
    assert md.startswith("> **SIMULATION")
    assert "H2b" in md and "\\|" in md
    for line in md.splitlines():                                    # chaque ligne du tableau des verdicts a 4 cellules
        if line.startswith("| H") and "Instrument" not in line and line.count("\\|") == 0:
            assert line.count("|") == 5, line
    assert "Runs du plan sans résultat (22/25)" in md              # 3 runs simulés sur 25
    real = RP.render_report(recs, ver, simulated=False)
    assert "SIMULATION" not in real


def test_is_simulated_and_missing_runs(sim):
    assert RP.is_simulated(sim["root"] / "runs")
    assert RP.missing_runs(R.build_plan(), sim["root"] / "runs") == [r.run_id for r in R.build_plan() if r.run_id not in sim["ids"]]
    assert not RP.is_simulated(sim["root"] / "vide")


# ----------------------------------------------------------------------------- scripts
def test_plan_script_json_and_budget_guard(tmp_path):
    r = run_script("plan.py", "--steps", "1000", "--json", cwd=tmp_path)
    assert r.returncode == 0
    plan = json.loads(r.stdout)
    assert len(plan) == 25 and plan[0]["steps"] == 1000 and plan[0]["images_seen"] == 64_000
    assert all(5e6 < e["params"] < 7e6 for e in plan)               # budget de paramètres ~ 6 M pour les trois familles
    r = run_script("plan.py", "--budget-hours", "38", cwd=tmp_path)
    assert r.returncode != 0 and "mesuré" in (r.stdout + r.stderr)  # pas de budget sans temps par pas mesuré


def test_plan_script_with_timing(tmp_path):
    timing = {"seconds_per_step": {"regression": 0.1, "oracle": 0.1, "diffusion": 0.1, "tokens-vq": 0.1, "tokens-gpt": 0.1}}
    (tmp_path / "timing.json").write_text(json.dumps(timing), encoding="utf-8")
    r = run_script("plan.py", "--steps", "3600", "--timing", "timing.json", cwd=tmp_path)
    assert "2.5 h de GPU" in r.stdout                               # 25 runs x 3600 pas x 0,1 s = 9 000 s = 2,5 h
    r = run_script("plan.py", "--budget-hours", "10", "--timing", "timing.json", cwd=tmp_path)
    assert "-> 14400 pas par run" in r.stdout                       # 10 h = 36 000 s ; 25 x 0,1 s = 2,5 s par pas
    r = run_script("plan.py", "--budget-hours", "10.1", "--timing", "timing.json", cwd=tmp_path)
    assert "-> 14500 pas par run" in r.stdout                       # 36 360 / 2,5 = 14 544 -> arrondi à la centaine inférieure


def test_validate_instrument_script(sim, tmp_path):
    r = run_script("validate_instrument.py", "--data", sim["root"] / "data", "--out", tmp_path / "inst.json", cwd=tmp_path)
    rep = json.loads((tmp_path / "inst.json").read_text(encoding="utf-8"))
    assert (r.returncode == 0) == rep["h0a"]["pass"]               # le code de sortie suit le verdict H0a
    assert rep["n_pairs"] == 200 and rep["avertissement_taille"]   # 200 paires < 600 : l'avertissement est écrit
    assert "paires seulement" in r.stdout
    assert 0.1 < rep["rho_max"] < 0.25
    r = run_script("validate_instrument.py", "--data", tmp_path / "vide", cwd=tmp_path)
    assert r.returncode != 0 and "generate_data" in (r.stdout + r.stderr)


def test_analyze_script_refuses_to_mix_simulation_into_reports(sim, tmp_path):
    r = run_script("analyze.py", "--runs", sim["root"] / "runs", "--out", "reports", "--no-figures", cwd=tmp_path)
    assert r.returncode != 0 and "SIMULÉS" in (r.stdout + r.stderr)
    assert not (tmp_path / "reports").exists()


def test_analyze_script_on_simulated_runs(sim, tmp_path):
    r = run_script("analyze.py", "--runs", sim["root"] / "runs", cwd=tmp_path)          # sortie par défaut : reports/dryrun
    assert r.returncode == 0, r.stderr
    out = tmp_path / "reports" / "dryrun"
    assert "SIMULATION" in r.stdout
    for f in ("verdicts.json", "report.md", "table_erreurs.csv", "calcul.csv", "fig_flou.png"):
        assert (out / f).exists(), f
    ver = json.loads((out / "verdicts.json").read_text(encoding="utf-8"))
    assert ver["_simulation"] is True and ver["H0a"]["status"] == "non évaluable"      # pas d'instrument fourni
    assert (out / "report.md").read_text(encoding="utf-8").startswith("> **SIMULATION")


def test_analyze_script_errors(tmp_path):
    r = run_script("analyze.py", "--runs", tmp_path / "rien", cwd=tmp_path)
    assert r.returncode != 0 and "introuvable" in (r.stdout + r.stderr)
    (tmp_path / "vide").mkdir()
    r = run_script("analyze.py", "--runs", tmp_path / "vide", cwd=tmp_path)
    assert r.returncode != 0 and "aucun eval.json" in (r.stdout + r.stderr)


def _fill(path, col, fn):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    for i, r in enumerate(rows):
        r[col] = fn(i, r)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_human_review_scripts_round_trip(sim, tmp_path):
    ids = sim["ids"]
    common = ["--runs", sim["root"] / "runs", "--data", sim["root"] / "data"]
    r = run_script("human_review.py", "review-make", *common, "--select", ids[0], f"{ids[1]}:n=3", f"{ids[2]}:tau=1",
                   "-n", "60", "--out", tmp_path / "rev", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / "rev").glob("review_page_*.png"))) == 6
    _fill(tmp_path / "rev" / "review.csv", "human_class", lambda i, row: row["auto_class"])    # relecteur d'accord
    r = run_script("human_review.py", "review-score", "--dir", tmp_path / "rev", cwd=tmp_path)
    assert r.returncode == 0 and "accord brut" in r.stdout
    s = json.loads((tmp_path / "rev" / "review_score.json").read_text(encoding="utf-8"))
    assert s["n_scored"] == 60 and s["accord_brut"]["value"] == 1.0
    # sans rien remplir : message clair, code non nul
    r = run_script("human_review.py", "review-make", *common, "--select", ids[0], "-n", "20", "--out", tmp_path / "rev2", cwd=tmp_path)
    r = run_script("human_review.py", "review-score", "--dir", tmp_path / "rev2", cwd=tmp_path)
    assert r.returncode != 0 and "aucune ligne remplie" in (r.stdout + r.stderr)


def test_ab_scripts_round_trip(sim, tmp_path):
    ids = sim["ids"]
    common = ["--runs", sim["root"] / "runs", "--data", sim["root"] / "data"]
    a, b = f"{ids[1]}:n=3", ids[0]
    r = run_script("human_review.py", "ab-make", *common, "--a", a, "--b", b, "--out", tmp_path / "ab", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    key = list(csv.DictReader(open(tmp_path / "ab" / "key.csv", newline="", encoding="utf-8")))
    for k in (1, 2, 3):                                              # les trois relecteurs préfèrent la diffusion
        side = {int(x["pair_id"]): ("L" if x["left"] == a else "R") for x in key}
        _fill(tmp_path / "ab" / f"reviewer_{k}.csv", "choice", lambda i, row: side[int(row["pair_id"])])
    r = run_script("human_review.py", "ab-score", "--dir", tmp_path / "ab", "--responses", "reviewer_1.csv", "reviewer_2.csv",
                   "reviewer_3.csv", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / "ab" / "ab_score.json").read_text(encoding="utf-8"))
    assert s["reponses_valides"] == 60
    assert s[f"preference_{a}_jugements"]["value"] == 1.0 or s[f"preference_{b}_jugements"]["value"] == 1.0
    assert "κ moyen" in r.stdout


# ----------------------------------------------------------------------------- données requises par le plan
def _tiny_data(root, ps, splits=("train", "val", "test")):
    for p in ps:
        for sp in splits:
            args = ["--split", sp, "--p", str(p), "--out", root]
            args += ["--n-bifurcations", "20"] if sp == "test" else ["--n-transitions", "200"]
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_data.py"), *map(str, args)],
                               capture_output=True, text=True, timeout=120)
            assert r.returncode == 0, r.stderr


def test_preflight_names_every_missing_file(tmp_path):
    plan = R.build_plan(steps=10)
    pre = R.preflight(plan, tmp_path / "data")
    # quatre valeurs de p (0 ; 0,1 ; 0,3 ; 0,5) x (train + test) manquants ; validation seulement en avertissement
    assert len(pre["manquants"]) == 8 and len(pre["avertissements"]) == 4
    assert any("p0.00" in m and "test_bifurcation" in m for m in pre["manquants"])   # le contrôle à p = 0 est couvert
    _tiny_data(tmp_path / "data", [0.0, 0.1, 0.3, 0.5])
    assert R.preflight(plan, tmp_path / "data") == {"manquants": [], "avertissements": []}


def test_preflight_notices_a_missing_test_set_only(tmp_path):
    _tiny_data(tmp_path / "data", [0.3], splits=("train", "val"))
    pre = R.preflight([r for r in R.build_plan(steps=10) if r.p == 0.3], tmp_path / "data")
    assert len(pre["manquants"]) == 1 and "test_bifurcation" in pre["manquants"][0] and not pre["avertissements"]


def test_run_plan_refuses_to_start_without_data_and_lists_status_with_it(tmp_path):
    r = run_script("run_plan.py", "--dry", "--no-eval", "--groups", "core", "control", "--data", tmp_path / "data",
                   "--runs", tmp_path / "runs", cwd=tmp_path)
    assert r.returncode != 0 and "données manquantes" in (r.stdout + r.stderr) and "p0.00" in (r.stdout + r.stderr)
    assert not (tmp_path / "runs").exists()                          # rien n'a été créé
    _tiny_data(tmp_path / "data", [0.0, 0.3])
    r = run_script("run_plan.py", "--dry", "--no-eval", "--groups", "core", "control", "--data", tmp_path / "data",
                   "--runs", tmp_path / "runs", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("à faire") == 12 and "[12/12]" in r.stdout


def test_run_plan_requires_instrument_before_evaluating(tmp_path):
    r = run_script("run_plan.py", "--groups", "core", "--data", tmp_path / "data", "--runs", tmp_path / "runs",
                   "--instrument", tmp_path / "absent.json", cwd=tmp_path)
    assert r.returncode != 0 and "validate_instrument" in (r.stdout + r.stderr)


def test_train_script_rejects_unknown_run_before_importing_torch(tmp_path):
    r = run_script("train.py", "--run-id", "nimporte-quoi", cwd=tmp_path)
    assert r.returncode != 0 and "run inconnu" in (r.stdout + r.stderr) and "core-regression-p0.30-s0" in (r.stdout + r.stderr)


def test_status_of_follows_the_files_present(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_plan
    d = tmp_path / "r"
    assert run_plan.status_of(d) == "à faire"
    d.mkdir()
    (d / "ckpt.pt").write_bytes(b"x")
    assert run_plan.status_of(d) == "en cours"
    (d / "final.pt").write_bytes(b"x")
    assert run_plan.status_of(d) == "entraîné"
    (d / "eval.json").write_text("{}")
    assert run_plan.status_of(d) == "évalué"


# ----------------------------------------------------------------------------- PyTorch absent : message clair
import importlib.util

no_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is not None,
                              reason="PyTorch est installé : ces scripts lanceraient un vrai entraînement")


@no_torch
@pytest.mark.parametrize("script,args", [("smoke_test.py", []), ("train.py", ["--run-id", "core-regression-p0.30-s0"]),
                                         ("train_probe.py", []), ("evaluate.py", ["--run-id", "x", "--rho-max", "0.16"])])
def test_torch_scripts_explain_how_to_install_pytorch(script, args, tmp_path):
    r = run_script(script, *args, cwd=tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "PyTorch est requis" in out and "uv sync --extra train" in out
    assert "Traceback" not in out


@no_torch
def test_run_plan_checks_pytorch_before_anything_else(tmp_path):
    r = run_script("run_plan.py", "--groups", "core", "--no-eval", "--data", tmp_path / "data", cwd=tmp_path)
    assert r.returncode != 0 and "PyTorch est requis" in (r.stdout + r.stderr)
    assert not (tmp_path / "runs").exists()
