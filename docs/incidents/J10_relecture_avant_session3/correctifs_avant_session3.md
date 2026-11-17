# Correctifs proposés avant la session 3 (relecture du commit 61dfcc6)

Contrôlé : les 271 tests NumPy passent ; la règle de pente rend J6 seul en alerte sur les journaux de la session 2 ;
la cellule de contrôle du notebook s'exécute sur un run simulé (lignes « au rail » comprises) ; `evaluate.py --all --force`
appelle `evaluate_run` avec les mêmes réglages que `run_plan.py` (graine 0, n = 1, 3, 5, 20), donc la réévaluation refait
la même mesure plus la clé `rail`.
**Non exécuté** : tout ce qui demande PyTorch, un GPU ou les poids (`sigma_sweep.py`, `evaluate.py`, l'entraînement,
le notebook sur Kaggle).

## 1. Le garde-fou « Dataset périmé » ne couvre pas le cas qui va se présenter (à corriger)

Le garde-fou de la cellule 1 ne lève l'erreur que si `core-regression-p0.30-s0` n'est pas évalué. Or le notebook exporté porte,
dans ses métadonnées, une version précise du Dataset (`sourceId` 22284860), celle attachée pendant la session 2 : d'après le
`ls runs/*/` de cette session, l'état repris alors ne contenait qu'**un** run, c'est donc l'état de la fin de la session 1
(je n'ai pas pu relier ce numéro à une version sur Kaggle ; je le déduis de ce `ls`). Une session 3 lancée sans « mise à jour de
l'entrée » passerait le garde-fou et recommencerait les runs 2 à 5 (environ 10 h de GPU).

Cellule 0, ajouter :

```python
MIN_EVALUATED = 5          # runs déjà évalués dans l'état qu'on s'attend à reprendre ; à relever après chaque session
```

Cellule 1, remplacer le garde-fou final par :

```python
n_evaluated = len(glob.glob(f'{BASE_DIR}/runs/*/eval.json'))
print(f"runs évalués dans l'état repris : {n_evaluated} (au moins {MIN_EVALUATED} attendus)")
if STAGE == 'core_control' and n_evaluated < MIN_EVALUATED:
    raise RuntimeError(f"Entrée flou-wm-etat périmée : {n_evaluated} runs évalués, {MIN_EVALUATED} attendus. "
                       "Mettre l'entrée sur sa dernière version avant de lancer la session.")
```

Testé sur un dossier à 1 run (erreur levée) et sur l'état de la session 2 (5 runs, passe).

## 2. Le balayage de σ : prédiction fausse et plage mal choisie (à corriger AVANT de le lancer)

`sigma_sweep.py` annonce qu'un pas depuis un σ « bien couvert » (1 à 5) doit redonner « flou élevé, α̂ resserré autour de p ».
Ce n'est pas ce que ferait un débruiteur **exact**. Le pas part de x = σ·n (du bruit pur). La moyenne a posteriori de
{A, B} sur ce x a un logit de moyenne logit p et d'écart-type d/σ, avec d = ‖A − B‖ ≈ 15 à l'échelle du modèle ([−1, 1]).
Dès que σ est inférieur à d, le bruit seul tranche : l'image sort **nette** (A ou B), pas floue, et ρ reste nul.
Référence calculée (NumPy, `ideal_sigma_reference.py`, 300 états de test, p = 0,3) :

| σ | floue | nettes | α̂ p5 – p95 | signal / bruit |
|---|---|---|---|---|
| 1 | 7 % | 93 % | 0,000 – 1,000 | 0,82 |
| 2 | 14 % | 86 % | 0,000 – 1,000 | 0,41 |
| 5 | 34 % | 66 % | 0,003 – 0,984 | 0,16 |
| 10 | 56 % | 44 % | 0,035 – 0,845 | 0,08 |
| 20 | 75 % | 25 % | 0,109 – 0,599 | 0,04 |
| 40 | 92 % | 8 % | 0,185 – 0,444 | 0,02 |
| 80 | 99,7 % | 0,3 % | 0,239 – 0,369 | 0,01 |

Deux conséquences.

- Le cours et la docstring ont raison à σ = 80 seulement : α̂ resserré (0,24 à 0,37), « autre » = 0. À σ = 1 à 5, la bonne
  attente est « nettes », pas « floue ». Lire le balayage contre cette table, pas contre « flou élevé ».
- Il n'existe pas de σ à la fois bien couvert à l'entraînement et dont le point de départ soit dans la loi d'entraînement.
  Le réseau a été entraîné sur « image + σ·bruit ». Les images cibles ont une moyenne de −0,77 par pixel : un bruit pur à
  σ = 1 s'en écarte de 86 écarts-types sur la moyenne, à σ = 10 de 9, à σ = 20 de 4, à σ = 80 de 1. Le départ n'est dans la loi
  qu'à σ ≳ 40 à 80, là où il y a 51 exemples d'entraînement (σ ≥ 40) ou 4 (σ ≥ 80), contre 21 000 à σ ≥ 5.
  Un mauvais résultat à σ ≤ 10 ne dirait donc pas « σ mal couvert » : il peut venir du point de départ.

À faire : ajouter des σ dans la zone utile (la CLI les accepte déjà, aucun changement de code), écrire cette table dans
`DECISIONS.md` comme lecture de référence **avant** de lancer, et lire « autre » et la dispersion de α̂ contre elle.

```python
!python scripts/sigma_sweep.py --all --sigmas 1 2 5 10 20 30 40 60 80
```

Observation (mon rapprochement, non vérifié sur le réseau) : le α̂ mesuré à n = 1 va de 0,03 à 0,86 (5e–95e centiles, J9),
ce que le débruiteur exact donne à σ ≈ 10 (0,035 – 0,845), pas à σ = 80. Le réseau se comporterait à σ = 80 comme s'il
voyait un bruit dix fois plus faible. À regarder dans `sigma_sweep.json`, sans en faire une conclusion.

## 3. Réévaluation idempotente (facultatif)

Avec `--all --force`, oublier de remettre `RUN_REEVAL = False` coûte 14 minutes et réécrit les `eval.json` à chaque session.
Variante qui ne réévalue que les runs finis dont l'`eval.json` n'a pas la clé `rail` (testée : elle désigne les cinq runs de
la session 2, et plus aucun une fois qu'ils portent la clé) :

```python
if RUN_REEVAL:
    finished = [os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob('runs/*/eval.json'))]
    stale = [r for r in finished if os.path.exists(f'runs/{r}/final.pt') and not all(
        'rail' in v['summary'] for v in json.load(open(f'runs/{r}/eval.json'))['variants'])]
    print("à réévaluer (sans clé rail) :", stale or "aucun")
    for run_id in stale:
        out = subprocess.run(['python', 'scripts/evaluate.py', '--run-id', run_id, '--force'],
                             capture_output=True, text=True)
        print(out.stdout[-2000:], out.stderr[-1500:])
```

## 4. Réserves à écrire dans le rapport

- **Seuil de ρ au rail non recalibré.** La mesure au rail réutilise ρmax = 0,1599, calibré sur l'image entière. Au rail,
  le même bruit (σ = 0,01) donne un quantile à 99,5 % de ρ de 0,036 (mesuré sur 300 états de p = 0,3), soit 4,4 fois moins : le
  rail est plus tolérant que la mesure principale, ses « autre » sont un minorant. `eval_arrays.npz` garde `__rho_rail` par
  état : on peut re-seuiller après coup, sans réévaluer.
- **« 17 minutes mesurées »** : 13,8 min sont mesurées (somme des `seconds` des cinq `eval.json` : 5,2 + 5,2 + 0,3 + 0,3 + 2,9) ;
  le balayage de σ n'a pas encore été chronométré.
- **README** : « 265 tests » devrait dire 272 (271 passés, 1 sauté sans `gymnasium` ici).
- **Réévaluation** : comparer les `eval.json` réécrits aux tableaux de J9 (p̂ 0,327 et 0,255, ratios MSE). Les nombres ne
  devraient pas bouger au-delà de l'indéterminisme GPU ; sinon, le dire.

## 5. Message à transmettre à l'agent

> Relecture du commit 61dfcc6 : règle de pente, mesure au rail, tests et cellule de contrôle vérifiés de mon côté, merci.
> Trois points avant la session 3. (1) Le garde-fou « Dataset périmé » ne teste que le premier run : une entrée restée sur la
> version de la session 1 passerait. Remplacer par un compte de `eval.json` contre `MIN_EVALUATED = 5` (code dans
> `correctifs_avant_session3.md`). (2) La docstring de `sigma_sweep.py` prédit « flou élevé à σ = 1 à 5 », ce qui est faux
> même pour un débruiteur exact (le bruit seul tranche la branche quand σ < ‖A − B‖ ≈ 15 : image nette, ρ = 0), et le point de
> départ x = σ·n n'est dans la loi d'entraînement que pour σ ≳ 40. Ajouter `scripts/ideal_sigma_reference.py`, écrire la table
> de référence dans `DECISIONS.md` avant de lancer, et lancer le balayage avec `--sigmas 1 2 5 10 20 30 40 60 80`.
> (3) Réserve à ajouter à la décision 3 : ρmax n'est pas recalibré au rail (quantile du bruit 0,036 au lieu de 0,164).
