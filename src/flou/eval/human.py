"""Outils de relecture humaine : relecture manuelle de 100 cas et test A/B en aveugle.

Relecture manuelle (dossier §1.9, cours chap. 7) : 100 cas tirés de façon STRATIFIÉE par classe automatique
(nette-A, nette-B, floue, autre), poolés sur les modèles évalués, pour estimer les erreurs du classifieur
contrefactuel. Le relecteur voit la prédiction et les deux images de référence A et B, et note
`human_class` dans un CSV. Le score donne la matrice de confusion, l'accord par classe avec son intervalle de
Clopper-Pearson, et une exactitude PONDÉRÉE par la prévalence des classes dans la population poolée (un
échantillon stratifié ne donne pas directement le taux d'erreur global).

Test A/B en aveugle : 20 paires issues du MÊME contexte, deux modèles, ordre gauche/droite aléatoire, 3
relecteurs indépendants. La clé (qui est à gauche) est écrite dans un fichier séparé, jamais montré aux
relecteurs. Le score rapporte la proportion de préférences (avec Clopper-Pearson), l'accord entre relecteurs
(κ de Cohen moyen sur les paires de relecteurs) et la lecture honnête : 20 paires donnent un intervalle large
(15 préférences sur 20 -> [0,51 ; 0,91]).

Pillow est requis pour écrire les images ; le score n'a besoin que de csv et NumPy.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..game.viz import contact_sheet
from .classifier import CLASS_NAMES
from .stats import cohen_kappa, mean_pairwise_kappa, proportion

HUMAN_ALIASES = {"a": 0, "nette-a": 0, "b": 1, "nette-b": 1, "floue": 2, "flou": 2, "blur": 2,
                 "autre": 3, "other": 3}


# ---------------------------------------------------------------------------------------
# Lecture des runs évalués (eval_arrays.npz, voir flou.eval.arrays)
# ---------------------------------------------------------------------------------------
def parse_selection(text: str) -> tuple[str, str]:
    """`run_id` ou `run_id:variante` (par exemple `core-diffusion-p0.30-s0:n=3`) -> (run_id, variante)."""
    run_id, _, variant = text.partition(":")
    return run_id, variant


def _arrays(runs_dir: Path, run_id: str) -> dict:
    f = Path(runs_dir) / run_id / "eval_arrays.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} manquant : le run doit être évalué (scripts/evaluate.py)")
    return dict(np.load(f))


def _test_set(runs_dir: Path, data_root: Path, run_id: str):
    from ..data.dataset import BifurcationSet
    from ..train.runs import RunSpec, test_file
    spec = RunSpec.load(Path(runs_dir) / run_id)
    return spec, BifurcationSet.load(test_file(data_root, spec.p))


def load_pool(runs_dir: str | Path, data_root: str | Path, selections: list[tuple[str, str]]) -> dict:
    """Réservoir de relecture : images prédites plafonnées par classe, avec les références A et B, poolées sur les
    variantes choisies ; `population` = classes de TOUTES les images prédites de ces variantes."""
    from .arrays import variant_key
    imgs, A, B, labels, alpha, rho, source, population = [], [], [], [], [], [], [], []
    for run_id, variant in selections:
        arr = _arrays(Path(runs_dir), run_id)
        key = variant_key(variant)
        if f"{key}__img" not in arr:
            raise KeyError(f"{run_id} : variante « {key} » sans images sauvegardées (variantes présentes : "
                           f"{sorted({k.split('__')[0] for k in arr})})")
        _, bset = _test_set(Path(runs_dir), Path(data_root), run_id)
        a_all, b_all, _ = bset.counterfactuals()
        idx = arr[f"{key}__img_idx"]
        st = arr[f"{key}__state"][idx]
        imgs.append(arr[f"{key}__img"])
        A.append(a_all[st])
        B.append(b_all[st])
        labels.append(arr[f"{key}__label"][idx])
        alpha.append(arr[f"{key}__alpha"][idx])
        rho.append(arr[f"{key}__rho"][idx])
        source += [f"{run_id}:{key}"] * len(idx)
        population.append(arr[f"{key}__label"])
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    return {"imgs": cat(imgs), "A": cat(A), "B": cat(B), "labels": cat(labels).astype(int), "alpha": cat(alpha),
            "rho": cat(rho), "source": source, "population": cat(population).astype(int)}


def load_ab_pair(runs_dir: str | Path, data_root: str | Path, sel_a: tuple[str, str], sel_b: tuple[str, str]) -> dict:
    """Prédictions des deux modèles sur les MÊMES contextes (les n_ab premiers états du jeu de test du même p)."""
    from .arrays import variant_key
    out, specs = [], []
    for run_id, variant in (sel_a, sel_b):
        arr = _arrays(Path(runs_dir), run_id)
        key = variant_key(variant)
        if f"{key}__ab_img" not in arr:
            raise KeyError(f"{run_id} : pas de {key}__ab_img (premier tirage rangé par état)")
        spec, bset = _test_set(Path(runs_dir), Path(data_root), run_id)
        out.append(arr[f"{key}__ab_img"])
        specs.append((spec, bset))
    if abs(specs[0][0].p - specs[1][0].p) > 1e-9:
        raise ValueError("les deux modèles doivent avoir le même p (mêmes états de test)")
    n = min(len(out[0]), len(out[1]))
    bset = specs[0][1]
    ctx_last = bset.contexts()[:n, -1]
    return {"pred_a": out[0][:n], "pred_b": out[1][:n], "ctx_last": ctx_last,
            "name_a": ":".join(sel_a).rstrip(":"), "name_b": ":".join(sel_b).rstrip(":")}


# ---------------------------------------------------------------------------------------
# Relecture manuelle
# ---------------------------------------------------------------------------------------
def stratified_sample(labels: np.ndarray, n: int = 100, rng: np.random.Generator | None = None,
                      n_classes: int = 4) -> np.ndarray:
    """n indices tirés par classe, à parts égales ; une classe trop petite est prise en entier et son manque
    est redistribué aux autres. Renvoie des indices triés."""
    rng = rng or np.random.default_rng(0)
    labels = np.asarray(labels)
    pools = {c: np.nonzero(labels == c)[0] for c in range(n_classes)}
    quota = {c: 0 for c in pools}
    remaining, open_ = n, [c for c in pools if len(pools[c]) > 0]
    while remaining > 0 and open_:
        share = max(1, remaining // len(open_))
        for c in list(open_):
            take = min(share, len(pools[c]) - quota[c], remaining)
            quota[c] += take
            remaining -= take
            if quota[c] >= len(pools[c]):
                open_.remove(c)
            if remaining == 0:
                break
    idx = np.concatenate([rng.choice(pools[c], quota[c], replace=False) for c in pools if quota[c] > 0])
    return np.sort(idx)


def write_review(out_dir: str | Path, imgs: np.ndarray, A: np.ndarray, B: np.ndarray, labels: np.ndarray,
                 alpha: np.ndarray, rho: np.ndarray, source: list[str], n: int = 100, seed: int = 0,
                 per_page: int = 10, population: np.ndarray | None = None) -> dict:
    """Écrit planches PNG + review.csv (à remplir) + review_meta.json (prévalences de la population poolée).

    imgs, A, B : uint8 (N, 64, 64, 3) ; labels, alpha, rho : classification automatique ; source : nom du
    modèle/variante de chaque ligne. Les planches montrent [prédiction | A | B] ; la classe automatique n'y
    figure PAS (la relecture est faite sans la voir).

    population : classes automatiques de TOUTE la population dont `imgs` est un réservoir (par défaut `labels`).
    Elle sert aux prévalences de l'exactitude pondérée : le réservoir est plafonné par classe, il ne reflète
    donc pas les proportions réelles."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = stratified_sample(labels, n, rng)
    idx = idx[rng.permutation(len(idx))]                         # ordre aléatoire : pas de blocs par classe
    rows = []
    for k, i in enumerate(idx):
        rows.append({"case_id": k + 1, "source": source[i], "state": int(i), "auto_class": CLASS_NAMES[int(labels[i])],
                     "alpha": f"{alpha[i]:.3f}", "rho": f"{rho[i]:.3f}", "human_class": "", "notes": ""})
    for page, lo in enumerate(range(0, len(idx), per_page)):
        sel = idx[lo:lo + per_page]
        sheet = [[imgs[i], A[i], B[i]] for i in sel]
        lab = [[f"cas {lo + j + 1} : prediction", "A (reference)", "B (reference)"] for j in range(len(sel))]
        contact_sheet(sheet, out / f"review_page_{page + 1:02d}.png", scale=3, labels=lab)
    with open(out / "review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pop = np.asarray(labels if population is None else population)
    counts = {name: int((pop == c).sum()) for c, name in enumerate(CLASS_NAMES)}
    meta = {"n_cases": len(rows), "population": counts, "n_population": int(len(pop)), "seed": seed,
            "consigne": "Pour chaque cas, noter human_class parmi : A, B, floue, autre. "
                        "A / B : l'image prédite est nette et identique à la référence correspondante ; "
                        "floue : mélange visible des deux ; autre : tout le reste (objet absent, déplacé, artefacts)."}
    (out / "review_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _parse_class(s: str) -> int | None:
    return HUMAN_ALIASES.get(s.strip().lower()) if s and s.strip() else None


def score_review(csv_path: str | Path, meta_path: str | Path | None = None, conf: float = 0.95) -> dict:
    """Compare la classe automatique et la classe humaine ; ignore (et compte) les lignes non remplies."""
    csv_path = Path(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    auto, human, bad = [], [], 0
    for r in rows:
        h = _parse_class(r["human_class"])
        if h is None:
            bad += 1
            continue
        auto.append(CLASS_NAMES.index(r["auto_class"]))
        human.append(h)
    auto, human = np.array(auto), np.array(human)
    if len(auto) == 0:
        return {"n_scored": 0, "n_blank": bad}
    conf_mat = np.zeros((4, 4), dtype=int)
    for a, h in zip(auto, human):
        conf_mat[a, h] += 1
    agree = proportion(int((auto == human).sum()), len(auto), conf)
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        m = auto == c
        if m.any():
            per_class[name] = proportion(int((human[m] == c).sum()), int(m.sum()), conf).as_dict()
    out = {"n_scored": int(len(auto)), "n_blank": bad, "confusion_auto_x_humain": conf_mat.tolist(),
           "classes": list(CLASS_NAMES), "accord_brut": agree.as_dict(), "accord_par_classe": per_class,
           "kappa_auto_humain": cohen_kappa(auto, human)}
    meta_path = Path(meta_path) if meta_path else csv_path.with_name("review_meta.json")
    if meta_path.exists():
        pop = json.loads(meta_path.read_text(encoding="utf-8"))["population"]
        tot = sum(pop.values())
        w = {n: pop[n] / tot for n in pop}
        acc = sum(w[n] * per_class[n]["value"] for n in per_class if n in w)
        norm = sum(w[n] for n in per_class if n in w)
        out["exactitude_ponderee_par_prevalence"] = acc / norm if norm else float("nan")
        out["prevalence_population"] = w
    return out


# ---------------------------------------------------------------------------------------
# Test A/B en aveugle
# ---------------------------------------------------------------------------------------
def make_ab(out_dir: str | Path, pred_a: np.ndarray, pred_b: np.ndarray, ctx_last: np.ndarray, name_a: str,
            name_b: str, n_pairs: int = 20, n_reviewers: int = 3, seed: int = 0) -> dict:
    """pred_a, pred_b : uint8 (N, 64, 64, 3), UNE prédiction par modèle pour chacun des N mêmes contextes.

    Écrit pairs/pair_XX.png (contexte | gauche | droite), reviewer_k.csv (colonne `choice` à remplir par L ou
    R) et key.csv (modèle à gauche / à droite) : la clé ne doit pas être remise aux relecteurs."""
    if pred_a.shape != pred_b.shape or len(pred_a) < n_pairs:
        raise ValueError("deux jeux de prédictions de même forme, avec au moins n_pairs contextes")
    out = Path(out_dir)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pred_a), n_pairs, replace=False))
    a_left = rng.random(n_pairs) < 0.5
    key = []
    for k, i in enumerate(idx):
        left, right = (pred_a[i], pred_b[i]) if a_left[k] else (pred_b[i], pred_a[i])
        contact_sheet([[ctx_last[i], left, right]], out / "pairs" / f"pair_{k + 1:02d}.png", scale=5,
                      labels=[["contexte", "GAUCHE (L)", "DROITE (R)"]])
        key.append({"pair_id": k + 1, "state": int(i), "left": name_a if a_left[k] else name_b,
                    "right": name_b if a_left[k] else name_a})
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    for r in range(1, n_reviewers + 1):
        with open(out / f"reviewer_{r}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pair_id", "choice"])
            w.writerows([[k + 1, ""] for k in range(n_pairs)])
    (out / "CONSIGNE.txt").write_text(
        "Pour chaque paire, l'image du haut-gauche est le contexte (instant courant). Les deux autres sont deux "
        "prédictions de l'instant suivant.\nQuestion : laquelle est la plus nette ET la plus plausible ?\n"
        "Répondre L (gauche) ou R (droite) dans la colonne `choice` de votre fichier reviewer_k.csv. "
        "Ne pas consulter key.csv. Ne pas se concerter avec les autres relecteurs.\n", encoding="utf-8")
    return {"n_pairs": n_pairs, "models": [name_a, name_b], "a_left_share": float(a_left.mean())}


def score_ab(key_csv: str | Path, response_csvs: list[str | Path], conf: float = 0.95) -> dict:
    """Préférences par modèle, accord entre relecteurs, lecture par paire (majorité)."""
    with open(key_csv, newline="", encoding="utf-8") as f:
        key = {int(r["pair_id"]): r for r in csv.DictReader(f)}
    models = sorted({m for r in key.values() for m in (r["left"], r["right"])})
    if len(models) != 2:
        raise ValueError("la clé doit désigner exactement deux modèles")
    pair_ids = sorted(key)
    votes = np.full((len(response_csvs), len(pair_ids)), -1, dtype=int)     # 1 si le relecteur choisit models[0]
    n_left = np.zeros(len(response_csvs), dtype=int)                        # réponses « L » par relecteur
    n_answered = np.zeros(len(response_csvs), dtype=int)
    for ri, path in enumerate(response_csvs):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                c = r["choice"].strip().upper()
                if c not in ("L", "R"):
                    continue
                n_answered[ri] += 1
                n_left[ri] += int(c == "L")
                k = key[int(r["pair_id"])]
                chosen = k["left"] if c == "L" else k["right"]
                votes[ri, pair_ids.index(int(r["pair_id"]))] = int(chosen == models[0])
    valid = votes >= 0
    n_valid, n_expected = int(valid.sum()), votes.size
    k0 = int((votes[valid] == 1).sum())
    pooled = proportion(k0, n_valid, conf) if n_valid else None
    per_reviewer = [float(votes[i][valid[i]].mean()) if valid[i].any() else float("nan") for i in range(len(votes))]
    full = valid.all(axis=0)                                                # paires jugées par tous
    maj = (votes[:, full].mean(axis=0) > 0.5) if full.any() else np.array([])
    ties = int((votes[:, full].mean(axis=0) == 0.5).sum()) if full.any() else 0
    by_pair = proportion(int(maj.sum()), int(len(maj)), conf) if len(maj) else None
    kappa = mean_pairwise_kappa(votes[:, full]) if full.any() and len(votes) > 1 else float("nan")
    return {"modeles": models, "reponses_valides": n_valid, "reponses_attendues": n_expected,
            f"preference_{models[0]}_jugements": pooled.as_dict() if pooled else None,
            f"preference_{models[0]}_par_relecteur": per_reviewer,
            f"preference_{models[0]}_majorite_par_paire": by_pair.as_dict() if by_pair else None,
            "egalites_a_la_majorite": ties, "kappa_moyen_entre_relecteurs": kappa,
            "part_de_reponses_L_par_relecteur": [float(a / b) if b else float("nan") for a, b in zip(n_left, n_answered)],
            "avertissement": "Les 60 jugements (20 paires x 3 relecteurs) ne sont pas indépendants : les paires "
                             "sont l'unité statistique. Préférer l'intervalle par paire ; 20 paires donnent un "
                             "intervalle large. L'ordre gauche/droite est le même pour les trois relecteurs : un "
                             "biais de position (part de L très éloignée de 0,5) fabrique de l'accord apparent ; "
                             "vérifier part_de_reponses_L_par_relecteur avant d'interpréter le κ."}
