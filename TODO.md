# Ce qu'il reste à faire

État au 16 novembre 2026. Tout le code NumPy (jeu, bots, données, instruments, analyse, relecture humaine, et l'extension
« décision » : lecteur de contact, planificateur, contrôles, verdicts) est exécuté et testé : 272 tests, 271 réussis et 1 ignoré
faute de Gymnasium sous Linux ; 270 réussis et 2 ignorés sous Windows 11 avec `PYTHONUTF8=1` (le second ignoré est le test de
pseudo-terminal, réservé à Unix). **Sans `PYTHONUTF8=1`, 6 tests échouent sous Windows** : voir le bloc D. Le code PyTorch tourne désormais, et **le premier résultat est acquis** : `core-regression-p0.30-s0` est entraîné et évalué,
H0b réussi (MSE / plancher de Bayes 1,011) et flou 98,2 %. Un premier essai avait été rejeté pour surapprentissage et les
données portées de 100 000 à 400 000 transitions (`docs/JOURNAL.md`, J6). **Les 11 autres runs prioritaires restent à
faire** : aucune hypothèse comparant les familles n'est donc encore évaluable.

**Ce qui demande vraiment un GPU** : A2, A6, A10 et A11 (et A4, facultatif). A1 n'est qu'une installation, A3, A5, A7 et A9 tournent
sur un simple processeur — mais il est plus simple de les lancer là où sont les runs et les données.
**Sur Kaggle, tout cela est déjà enchaîné par `notebooks/flou_kaggle.ipynb`**, dans cet ordre, en une session qui tourne en
arrière-plan (`docs/JOURNAL.md`, J8). Le notebook Colab fait de même mais exige que la machine reste éveillée.

Chemin critique : **A1 → A2 → A3 → A5 → A6 → A7 → B**, puis, pour l'extension « décision » : **A9 → A10 → A11 → A7**.
Les décisions du bloc C se prennent **avant A6** (C6 à C8 : avant A10) : les paramètres du jeu et du
plan se figent au premier entraînement, et toute modification ultérieure invalide les mesures déjà faites.

## A. Sur une machine avec GPU (bloquant, dans l'ordre)

- [ ] **A1. Installer.** Sur Colab, rien à faire : le notebook clone le dépôt et complète avec `pip` (PyTorch y est déjà).
  Sur une machine à vous : `uv sync --extra train` (ou `make sync-train`). `uv.lock` est **déjà versionné** (résolution PyPI faite le
  15 novembre, PyTorch compris) : l'installation doit donc reprendre ces versions figées. Si le téléchargement de la roue PyTorch
  échoue, copier l'erreur (c'est `pyproject.toml` qu'il faudra corriger).
  Ne pas relancer ensuite un `uv sync` nu : il désinstalle PyTorch (README §2).
- [ ] **A2. Test de fumée.** `make smoke` (`uv run scripts/smoke_test.py --time`). **Obligatoire, jamais exécuté** : s'attendre à des
  erreurs de forme de tenseurs ou d'API PyTorch à corriger. Corriger jusqu'à ce que tout affiche `[OK]` ; ne **pas** lancer le plan
  tant qu'un `[FAIL]` reste. Produit `reports/timing.json`.
- [ ] **A3. Données et instrument.** `make data`, puis `make validate`. Le contrôle **H0a** doit passer (il demande au moins
  600 paires de test, 1 000 par défaut) ; sinon ne mesurer aucun modèle avant d'avoir compris pourquoi.
- [ ] **A4. Sonde de position (facultatif).** Par défaut, la sonde de position des rollouts est analytique. Une sonde CNN apprise existe
  (`uv run scripts/train_probe.py`, puis `--probe-cnn` dans `evaluate.py`) ; si vous l'utilisez, la comparer d'abord à l'analytique
  (consigne dans l'en-tête de `scripts/evaluate.py`). Jamais exécutée, comme tout le code PyTorch.
- [ ] **A5. Budget.** `uv run scripts/plan.py --timing reports/timing.json --budget-hours 38` : nombre de pas par run qui tient dans le
  budget. Le « ≈ 1,5 h par run » du cours **n'a jamais été mesuré** ; la valeur par défaut (20 000 pas × lot 64) non plus.
- [ ] **A6. Les 25 entraînements.** `uv run scripts/run_plan.py --steps N` (N issu de A5). Reprenable : relancer la même commande après
  une coupure. Ordre de priorité : cœur (9) et contrôles (3), puis axe K (6), grille (1), axe p (6, sacrifié en premier).
- [ ] **A7. Analyse.** `make analyze` → `reports/report.md`. Avant de conclure sur H2b, regarder la courbe p̂(n)
  (`calibration_par_n.csv`, `fig_calibration.png`) : voir C2.
- [x] **A8. Notebooks.** `notebooks/flou_colab.ipynb` a été exécuté jusqu'au premier entraînement réel (J6), puis remplacé par
  `notebooks/flou_kaggle.ipynb` pour la campagne (J8). Reste à confirmer sur Kaggle, en mode commit : la publication de l'état
  vers le Dataset et le verdict de surapprentissage du run relancé.

### Extension « décision » (H4, H5 ; `docs/EXTENSION_DECISION.md`)

- [ ] **A9. Contrôles sans GPU.** `make validate-decision` (5 min sur 2 cœurs Linux, 14 min mesurées sous Windows 11). Crée
  `data/p0.30/test_decision.npz` et `reports/decision_instrument.json`. Exécuté deux fois, sur Linux puis sous Windows, avec des
  résultats identiques : H0e réussi (100 % d'accord sur 30 000 images, 887 contacts), H0f réussi de peu (0,16 contact par partie ;
  +3,16 points sur le bot aléatoire pour un seuil de 3). À relancer chez vous : les fichiers produits ne sont pas livrés.
  **Sous Windows : préfixer `PYTHONUTF8=1`**, sinon le script plante en affichant son résumé — le rapport est déjà écrit, mais le
  code de sortie non nul fait croire à un contrôle échoué.
- [ ] **A10. Niveau 1 (GPU, court).** Après A6 : `make decision` (`uv run scripts/evaluate_decision.py --all`). **Jamais exécuté** ;
  le test de fumée (A2) fait tourner ce code en tout petit : regarder sa ligne « extension décision ».
- [ ] **A11. Niveau 2 (GPU, long).** **Durée mesurée le 15 novembre** par le test de fumée sur T4 : aux réglages par défaut,
  1,0 h par run de régression, **11,7 h par run de diffusion**, 66 h pour les jetons (d'où leur exclusion). Les réglages ont donc
  été réduits à `--games 20 --candidates 32` pour tous les runs (≈ 7,6 h au total) : voir la révision de C7 dans
  `docs/DECISIONS.md`. Relancer H0f à ces réglages avant de mesurer. Pour chronométrer soi-même :
  `uv run scripts/evaluate_decision.py --all --level 2 --games 4 --game-steps 10 --no-save`, et lire l'estimation affichée par
  `smoke_test.py --time` (durée par run, réglages par défaut). Puis `uv run scripts/evaluate_decision.py --all --level 2`
  (6 runs : régression et diffusion du cœur). **Durée inconnue**, à ajouter aux ≈ 38 h du plan ; si elle est trop longue, réduire
  `--games` ou `--candidates` pour **tous** les runs, et le noter. Relancer ensuite `make analyze` (A7).

## B. Avec des humains (après l'analyse)

- [ ] **Relecture de 100 cas** pour estimer les erreurs du classifieur, et **test A/B en aveugle** (20 paires × 3 relecteurs) :
  `uv run scripts/human_review.py --help`. Il faut trouver trois relecteurs. Protocole et limites : `docs/EXPERIMENTS.md` §6.

## C. Décisions prises (avant A6)

**Toutes tranchées le 15 novembre 2026, avant tout entraînement : [`docs/DECISIONS.md`](docs/DECISIONS.md).** Ce document rappelle
l'hypothèse concernée, la décision et ses raisons, et fixe la règle de révision (après le premier run, toute modification s'ajoute
à sa date, sans effacer la précédente). Résumé :

- [x] **C1. Où est le « modèle du monde » ?** L'extension « décision » est implémentée (risque estimé, puis planification).
- [x] **C2. H2b.** Seuil gardé, n = 3 inchangé ; le verdict se lit avec la courbe p̂(n) (`fig_calibration.png`), gratuite puisque
  l'échantillonnage ne demande aucun ré-entraînement. Une réfutation sera lue comme « 3 pas d'Euler ne suffisent pas à retrouver p ».
- [x] **C3. Les sept choix d'interprétation** sont confirmés, avec deux réserves à écrire dans le rapport : ρmax ≈ 0,16 est une
  convention (la part de « autre » n'est pas un taux absolu ; joindre la table de sensibilité), et l'axe K fait varier aussi la
  capacité (5,91 M à K = 64 contre 6,59 M à K = 1 024).
- [x] **C4. Puissance.** 3 graines acceptées ; **k = 3 comparaisons simultanées fixé d'avance** (H1, H2a, H3a), soit un seuil de
  3,23 écarts-types poolés. Budget résiduel éventuel → graines supplémentaires sur l'axe K, pas sur l'axe p.
- [x] **C5. Budget.** 19 runs ≈ 28 h engagés (cœur, contrôles, axe K, grille) + ~10 h de réserve pour le niveau 2 ;
  `--skip-groups p_axis` dès le départ, l'axe p seulement en bonus. Chiffre ferme après A2 et A5.
- [x] **C6. H5a** gardée telle quelle, lecteur doux (le plus sévère) ; le niveau 2 n'est pas doublé avec le lecteur à seuil, dont le
  contraste est déjà payé au niveau 1 par H4a/H4b.
- [x] **C7. Seuils de l'extension** confirmés ; la marge fine de H0f (+3,16 pour un minimum de 3) est reproductible et non
  chanceuse — mêmes chiffres sur deux systèmes d'exploitation, et marge plus large sur une seconde graine de parties (+4,02 points,
  0,14 contact). Le planificateur est stable (score 1,10 puis 1,02) ; c'est le bot aléatoire de référence qui varie (−2,06 puis −3,00).
- [x] **C8. Récompense lue dans les pixels** assumée : elle rend tout écart du niveau 2 imputable au seul modèle de dynamique, et
  interdit en retour toute conclusion sur un système bout-en-bout.

## D. Non vérifié (à tester quand l'occasion se présente)

- [ ] **uv** : le vrai téléchargement PyPI et la suite de tests sont désormais vérifiés sous Windows 11 avec Python 3.12 (uv 0.11.2),
  et `uv.lock` est versionné. Reste la **vraie roue PyTorch** (CUDA ou CPU), macOS, et les autres versions de Python
  (`requires-python >= 3.10` est déclaré, vérifié seulement en 3.12/3.13). Si PyTorch n'a pas de roue pour votre Python :
  `uv sync --extra train --python 3.12`.
- [x] **Gymnasium** (fait le 15 novembre) : `check_env` a tourné sur Colab avec Gymnasium 1.4 et passe. Seul avertissement, sans
  conséquence : l'environnement n'ayant pas de `spec` (instancié directement, pas via `gymnasium.make`), le vérificateur saute le
  contrôle facultatif des modes de rendu. Pour le lever un jour, il faudrait enregistrer l'environnement avec `gymnasium.register`.
- [ ] **Terminaux et encodage.** Mesuré le 15 novembre sous Windows 11 (Python 3.12) : Python y écrit sa sortie en `cp1252`, donc tout
  script qui affiche des accents, `δ`, `ρ` ou les cadres du jeu plante sur un `UnicodeEncodeError` (`play.py`,
  `validate_instrument.py`, `validate_decision.py`, `human_review.py`, `pytest`). Les rapports JSON sont écrits avant l'affichage,
  mais le code de sortie non nul fait croire à un échec de contrôle. Contournement vérifié : préfixer `PYTHONUTF8=1` (tout passe).
  **Correctif durable à décider** : forcer l'UTF-8 une fois pour toutes dans `scripts/_bootstrap.py`
  (`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`, idem `stderr`) — deux lignes, mais c'est du code.
  Reste à essayer : le clavier interactif (`msvcrt`) dans Windows Terminal, et Terminal.app sous macOS (`--colors 256`).
- [x] **Test du pseudo-terminal sous Jupyter/Colab** (corrigé le 15 novembre) : `shutil.get_terminal_size()` donne la priorité à
  `COLUMNS` et `LINES` sur la taille réelle, et Colab les exporte à 80x24 ; le jeu refusait donc de démarrer dans un pseudo-terminal
  pourtant réglé à 100x50. Le test nettoie maintenant ces deux variables pour le sous-processus.
- [ ] **Extension « décision », côté PyTorch** : `evaluate_decision_run`, `plan_run`, le découpage des gros lots dans
  `TorchPredictor` (`max_batch = 512`, à baisser si la mémoire du GPU manque) et le débit d'inférence du test de fumée n'ont jamais
  tourné. Les jetons sont exclus du niveau 2 par défaut (coût) : `--families regression diffusion tokens` pour les inclure.
- [ ] **Extension « décision », robustesse** : deux graines de parties mesurées (50 parties chacune, graines 0 et 1), planificateur
  stable et H0f réussi aux deux ; une troisième n'apporterait probablement rien. Le lecteur de contact suppose en revanche la palette
  par défaut (sur le rail, seul l'ennemi est rouge) : H0e le revérifie si vous changez les couleurs.
- [ ] **Suite de tests avec PyTorch** : quelques tests (ceux qui vérifient le message « PyTorch est requis ») sont ignorés quand PyTorch
  est installé ; c'est normal. Relancer `make test` sur la machine GPU pour voir le décompte réel.

## E. Facultatif

- [ ] Pousser sur GitHub (`git init -b main`, puis `git remote add origin …`) : `uv.lock` est versionné, et `.gitignore` écarte
  `data/`, `runs/`, `reports/` et les sorties SIMULÉES (`data_dryrun/`, `runs_dryrun/`).
- [ ] Épingler une version de Python (`uv python pin 3.12`) une fois la compatibilité de PyTorch connue.
- [ ] Ajouter un linter standard (ruff) : `scripts/lint_names.py` ne vérifie que les noms, ce qui était faute de mieux pour du code
  PyTorch non exécutable.
- [ ] Intégration continue (`uv sync` puis `uv run pytest`, sur CPU) si le dépôt est hébergé.
- [ ] Le dossier initial compte 19 entraînements (≈ 28 h) ; le cours en compte 25 (≈ 38 h) et fait foi. Si le plan réduit est retenu,
  le dire explicitement aux étudiants.

## Ce que je peux faire ensuite

Corriger le code PyTorch dès que vous me renvoyez les traces du test de fumée (A2), corriger `pyproject.toml` si la résolution
échoue (A1), changer l'échantillonneur (C2), reformuler H5a ou ajouter un planificateur CEM (C6, C7).
