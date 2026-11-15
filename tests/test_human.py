"""Outils de relecture humaine : échantillon stratifié, feuille de relecture, test A/B en aveugle.

Les « relecteurs » ci-dessous sont des fonctions déterministes : on vérifie que le score retrouve ce qu'on y a
mis (accord parfait, erreurs planifiées, préférence planifiée), pas qu'un humain juge bien.
"""
import csv
import json

import numpy as np
import pytest

from flou.eval import human
from flou.eval.classifier import CLASS_NAMES


def _imgs(n, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (n, 64, 64, 3), dtype=np.uint8)


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_col(path, col, values):
    rows = _read(path)
    for r, v in zip(rows, values):
        r[col] = v
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- échantillon stratifié
def test_stratified_sample_is_balanced_and_unique():
    labels = np.repeat([0, 1, 2, 3], [400, 300, 200, 100])
    idx = human.stratified_sample(labels, 100, np.random.default_rng(0))
    assert len(idx) == 100 and len(set(idx.tolist())) == 100 and (np.diff(idx) > 0).all()
    assert np.bincount(labels[idx], minlength=4).tolist() == [25, 25, 25, 25]


def test_stratified_sample_redistributes_a_small_class():
    labels = np.repeat([0, 1, 2, 3], [500, 400, 90, 10])          # « autre » n'a que 10 cas
    idx = human.stratified_sample(labels, 100, np.random.default_rng(1))
    counts = np.bincount(labels[idx], minlength=4)
    assert counts[3] == 10 and counts.sum() == 100                # classe rare prise en entier
    assert counts[2] <= 90 and counts[0] >= 25 and counts[1] >= 25


def test_stratified_sample_with_an_empty_class_and_few_cases():
    labels = np.repeat([0, 2], [30, 20])                           # nette-B et autre n'existent pas
    idx = human.stratified_sample(labels, 100, np.random.default_rng(2))
    assert len(idx) == 50                                          # tout ce qui existe, pas plus


def test_stratified_sample_is_reproducible():
    labels = np.random.default_rng(5).integers(0, 4, 1000)
    a = human.stratified_sample(labels, 100, np.random.default_rng(7))
    b = human.stratified_sample(labels, 100, np.random.default_rng(7))
    assert (a == b).all()


# ----------------------------------------------------------------------------- relecture manuelle
@pytest.fixture()
def review(tmp_path):
    n = 400
    rng = np.random.default_rng(0)
    labels = rng.choice(4, n, p=[0.4, 0.3, 0.2, 0.1])
    alpha = rng.random(n)
    rho = rng.random(n) * 0.3
    imgs, A, B = _imgs(n, 1), _imgs(n, 2), _imgs(n, 3)
    src = [f"modele_{i % 3}" for i in range(n)]
    meta = human.write_review(tmp_path / "rev", imgs, A, B, labels, alpha, rho, src, n=100, seed=0)
    return {"dir": tmp_path / "rev", "labels": labels, "meta": meta}


def test_write_review_outputs(review):
    d = review["dir"]
    rows = _read(d / "review.csv")
    assert len(rows) == 100 and [int(r["case_id"]) for r in rows] == list(range(1, 101))
    assert all(r["human_class"] == "" for r in rows)
    assert len(list(d.glob("review_page_*.png"))) == 10
    meta = json.loads((d / "review_meta.json").read_text(encoding="utf-8"))
    assert sum(meta["population"].values()) == meta["n_population"] == 400
    assert meta["population"]["nette-A"] == int((review["labels"] == 0).sum())
    # l'ordre des cas est mélangé : pas de blocs de classes automatiques consécutifs sur toute la feuille
    auto = [r["auto_class"] for r in rows]
    assert len(set(auto[:10])) > 1


def test_score_review_perfect_agreement(review):
    d = review["dir"]
    rows = _read(d / "review.csv")
    _write_col(d / "review.csv", "human_class", [r["auto_class"] for r in rows])
    s = human.score_review(d / "review.csv")
    assert s["n_scored"] == 100 and s["n_blank"] == 0
    assert s["accord_brut"]["value"] == 1.0 and s["kappa_auto_humain"] == pytest.approx(1.0)
    assert s["exactitude_ponderee_par_prevalence"] == pytest.approx(1.0)
    assert np.trace(np.array(s["confusion_auto_x_humain"])) == 100


def test_score_review_planted_errors_and_weighting(review):
    """Le relecteur se trompe sur la moitié des cas « floue » (il note « autre »), jamais ailleurs."""
    d = review["dir"]
    rows = _read(d / "review.csv")
    fl = [i for i, r in enumerate(rows) if r["auto_class"] == "floue"]
    wrong = set(fl[: len(fl) // 2])
    vals = [("autre" if i in wrong else r["auto_class"]) for i, r in enumerate(rows)]
    _write_col(d / "review.csv", "human_class", vals)
    s = human.score_review(d / "review.csv")
    assert s["accord_par_classe"]["floue"]["value"] == pytest.approx(1 - len(wrong) / len(fl))
    assert s["accord_par_classe"]["nette-A"]["value"] == 1.0
    cm = np.array(s["confusion_auto_x_humain"])
    assert cm[CLASS_NAMES.index("floue"), CLASS_NAMES.index("autre")] == len(wrong)
    # exactitude pondérée : chaque classe pèse sa prévalence dans la POPULATION, pas dans l'échantillon stratifié
    prev = s["prevalence_population"]
    expected = sum(prev[n] * s["accord_par_classe"][n]["value"] for n in s["accord_par_classe"])
    assert s["exactitude_ponderee_par_prevalence"] == pytest.approx(expected / sum(prev.values()))
    assert s["accord_brut"]["value"] < 1.0 < s["accord_brut"]["ci_hi"] + 1e-9 or s["accord_brut"]["ci_hi"] < 1.0


def test_score_review_accepts_aliases_and_counts_blanks(review):
    d = review["dir"]
    rows = _read(d / "review.csv")
    alias = {"nette-A": "A", "nette-B": "b", "floue": "Floue", "autre": "AUTRE"}
    vals = [alias[r["auto_class"]] for r in rows]
    vals[:7] = [""] * 7                                            # 7 lignes non remplies
    _write_col(d / "review.csv", "human_class", vals)
    s = human.score_review(d / "review.csv")
    assert s["n_scored"] == 93 and s["n_blank"] == 7 and s["accord_brut"]["value"] == 1.0


def test_score_review_all_blank(review):
    s = human.score_review(review["dir"] / "review.csv")
    assert s == {"n_scored": 0, "n_blank": 100}


# ----------------------------------------------------------------------------- A/B en aveugle
@pytest.fixture()
def ab(tmp_path):
    n = 60
    pa, pb, ctx = _imgs(n, 10), _imgs(n, 11), _imgs(n, 12)
    info = human.make_ab(tmp_path / "ab", pa, pb, ctx, "diffusion", "regression", n_pairs=20, n_reviewers=3, seed=4)
    return {"dir": tmp_path / "ab", "info": info}


def test_make_ab_outputs_and_blinding(ab):
    d = ab["dir"]
    assert len(list((d / "pairs").glob("pair_*.png"))) == 20
    key = _read(d / "key.csv")
    assert len(key) == 20 and {k["left"] for k in key} | {k["right"] for k in key} == {"diffusion", "regression"}
    assert all(k["left"] != k["right"] for k in key)
    assert 0 < sum(k["left"] == "diffusion" for k in key) < 20     # l'ordre est bien aléatoire
    for r in (1, 2, 3):
        rows = _read(d / f"reviewer_{r}.csv")
        assert list(rows[0]) == ["pair_id", "choice"] and len(rows) == 20 and all(x["choice"] == "" for x in rows)
        assert "diffusion" not in (d / f"reviewer_{r}.csv").read_text(encoding="utf-8")   # rien ne fuit
    assert "key.csv" in (d / "CONSIGNE.txt").read_text(encoding="utf-8")
    assert len({k["state"] for k in key}) == 20                    # 20 contextes distincts


def test_make_ab_is_reproducible_and_validates_inputs(tmp_path):
    pa, pb, ctx = _imgs(30, 1), _imgs(30, 2), _imgs(30, 3)
    human.make_ab(tmp_path / "x", pa, pb, ctx, "a", "b", seed=9)
    human.make_ab(tmp_path / "y", pa, pb, ctx, "a", "b", seed=9)
    assert (tmp_path / "x" / "key.csv").read_text(encoding="utf-8") == (tmp_path / "y" / "key.csv").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        human.make_ab(tmp_path / "z", pa, pb[:10], ctx, "a", "b")
    with pytest.raises(ValueError):
        human.make_ab(tmp_path / "w", pa[:5], pb[:5], ctx[:5], "a", "b", n_pairs=20)


def _vote(ab, reviewer, prefers):
    """prefers : fonction (ligne de clé) -> modèle préféré ; renvoie L/R dans le fichier du relecteur."""
    key = _read(ab["dir"] / "key.csv")
    ch = ["L" if prefers(k) == k["left"] else "R" for k in key]
    _write_col(ab["dir"] / f"reviewer_{reviewer}.csv", "choice", ch)
    return ab["dir"] / f"reviewer_{reviewer}.csv"


def test_score_ab_unanimous_preference(ab):
    files = [_vote(ab, r, lambda k: "diffusion") for r in (1, 2, 3)]
    s = human.score_ab(ab["dir"] / "key.csv", files)
    assert s["modeles"] == ["diffusion", "regression"]
    assert s["reponses_valides"] == 60
    pref = s["preference_diffusion_jugements"]
    assert pref["value"] == 1.0 and pref["k"] == 60 and pref["ci_lo"] > 0.9
    assert s["preference_diffusion_majorite_par_paire"]["value"] == 1.0
    assert "indépendants" in s["avertissement"] and "biais de position" in s["avertissement"]


def test_score_ab_position_bias_is_not_mistaken_for_preference(ab):
    """Un relecteur qui répond toujours L ne préfère aucun modèle : le résultat suit l'ordre aléatoire."""
    key = _read(ab["dir"] / "key.csv")
    left_diff = sum(k["left"] == "diffusion" for k in key)
    files = []
    for r in (1, 2, 3):
        _write_col(ab["dir"] / f"reviewer_{r}.csv", "choice", ["L"] * 20)
        files.append(ab["dir"] / f"reviewer_{r}.csv")
    s = human.score_ab(ab["dir"] / "key.csv", files)
    assert s["preference_diffusion_jugements"]["k"] == 3 * left_diff
    assert 0.0 < s["preference_diffusion_jugements"]["value"] < 1.0
    # piège : la même clé gauche/droite pour tous -> trois relecteurs biaisés vers L « s'accordent » parfaitement ;
    # c'est la part de L (tout contre 1) qui trahit le biais, pas le κ
    assert s["kappa_moyen_entre_relecteurs"] == pytest.approx(1.0)
    assert s["part_de_reponses_L_par_relecteur"] == [1.0, 1.0, 1.0]


def test_score_ab_planted_agreement_and_kappa(ab):
    """Relecteurs 1 et 2 préfèrent « diffusion » sur les 15 premières paires, « regression » sur les 5 dernières ;
    le 3 inverse tout ; préférence poolée 15+15+5 = 35 sur 60.
    κ(1,2) = 1 ; pour (1,3) et (2,3) : accord observé 0, accord attendu 0,75*0,25 + 0,25*0,75 = 0,375, donc
    κ = -0,375/0,625 = -0,6 (et non -1 : les marges ne sont pas équilibrées) ; moyenne = (1 - 0,6 - 0,6)/3."""
    r1 = _vote(ab, 1, lambda k: "diffusion" if int(k["pair_id"]) <= 15 else "regression")
    r2 = _vote(ab, 2, lambda k: "diffusion" if int(k["pair_id"]) <= 15 else "regression")
    r3 = _vote(ab, 3, lambda k: "regression" if int(k["pair_id"]) <= 15 else "diffusion")
    s = human.score_ab(ab["dir"] / "key.csv", [r1, r2, r3])
    assert s["preference_diffusion_jugements"]["k"] == 35 and s["reponses_valides"] == 60
    assert s["preference_diffusion_par_relecteur"] == pytest.approx([0.75, 0.75, 0.25])
    assert s["kappa_moyen_entre_relecteurs"] == pytest.approx((1 - 0.6 - 0.6) / 3)
    assert s["preference_diffusion_majorite_par_paire"]["k"] == 15        # majorité des 3 : 2 contre 1 sur les 20 paires
    assert s["egalites_a_la_majorite"] == 0


def test_score_ab_wide_interval_with_20_pairs(ab):
    """15 préférences sur 20 paires : l'intervalle de Clopper-Pearson reste large ([0,509 ; 0,913], valeur exacte
    confirmée avec scipy.stats.binomtest)."""
    files = [_vote(ab, r, lambda k: "diffusion" if int(k["pair_id"]) <= 15 else "regression") for r in (1, 2, 3)]
    s = human.score_ab(ab["dir"] / "key.csv", files)
    by_pair = s["preference_diffusion_majorite_par_paire"]
    assert (by_pair["k"], by_pair["n"]) == (15, 20)
    assert by_pair["ci_lo"] == pytest.approx(0.5090, abs=1e-3) and by_pair["ci_hi"] == pytest.approx(0.9134, abs=1e-3)


def test_score_ab_missing_answers_are_counted(ab):
    f1 = _vote(ab, 1, lambda k: "diffusion")
    f2 = _vote(ab, 2, lambda k: "diffusion")
    rows = _read(f2)
    for r in rows[:5]:
        r["choice"] = ""
    with open(f2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "choice"])
        w.writeheader()
        w.writerows(rows)
    s = human.score_ab(ab["dir"] / "key.csv", [f1, f2])
    assert s["reponses_valides"] == 35 and s["reponses_attendues"] == 40
    assert s["preference_diffusion_jugements"]["n"] == 35
    assert s["preference_diffusion_majorite_par_paire"]["n"] == 15        # seules les paires jugées par tous
