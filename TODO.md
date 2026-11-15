# Ce qu'il reste à faire

État au 15 novembre 2026. Tout le code NumPy (jeu, bots, données, instruments, analyse, relecture humaine, et l'extension
« décision » : lecteur de contact, planificateur, contrôles, verdicts) est exécuté et testé : 265 tests, 264 réussis et 1 ignoré
faute de Gymnasium sous Linux ; 263 réussis et 2 ignorés sous Windows 11 avec `PYTHONUTF8=1` (le second ignoré est le test de
pseudo-terminal, réservé à Unix). **Sans `PYTHONUTF8=1`, 6 tests échouent sous Windows** : voir le bloc D. **Aucun résultat scientifique n'existe encore** : le code PyTorch n'a jamais été exécuté et aucun entraînement n'a eu lieu.

**Ce que vous devez faire vous-même, parce qu'il faut un GPU** (je n'en ai pas) : A1, A2, A6, puis A9 à A11 pour l'extension.
Tout le reste du bloc A tourne sur un simple processeur.

Chemin critique : **A1 → A2 → A3 → A5 → A6 → A7 → B**, puis, pour l'extension « décision » : **A9 → A10 → A11 → A7**.
Les décisions du bloc C se prennent **avant A6** (C6 à C8 : avant A10) : les paramètres du jeu et du
plan se figent au premier entraînement, et toute modification ultérieure invalide les mesures déjà faites.

## A. Sur une machine avec GPU (bloquant, dans l'ordre)

- [ ] **A1. Installer.** `uv sync --extra train` (ou `make sync-train`). `uv.lock` est **déjà versionné** (résolution PyPI faite le
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
- [ ] **A8. Notebook Colab.** `notebooks/flou_colab.ipynb` n'a jamais été exécuté. À tester une fois si Colab est la voie choisie
  (il utilise pip, PyTorch y étant déjà installé). Sa première cellule **clone le dépôt GitHub** dans `/content/flou_wm` : vérifier
  l'URL (et y mettre un jeton si le dépôt est privé).

### Extension « décision » (H4, H5 ; `docs/EXTENSION_DECISION.md`)

- [ ] **A9. Contrôles sans GPU.** `make validate-decision` (5 min sur 2 cœurs Linux, 14 min mesurées sous Windows 11). Crée
  `data/p0.30/test_decision.npz` et `reports/decision_instrument.json`. Exécuté deux fois, sur Linux puis sous Windows, avec des
  résultats identiques : H0e réussi (100 % d'accord sur 30 000 images, 887 contacts), H0f réussi de peu (0,16 contact par partie ;
  +3,16 points sur le bot aléatoire pour un seuil de 3). À relancer chez vous : les fichiers produits ne sont pas livrés.
  **Sous Windows : préfixer `PYTHONUTF8=1`**, sinon le script plante en affichant son résumé — le rapport est déjà écrit, mais le
  code de sortie non nul fait croire à un contrôle échoué.
- [ ] **A10. Niveau 1 (GPU, court).** Après A6 : `make decision` (`uv run scripts/evaluate_decision.py --all`). **Jamais exécuté** ;
  le test de fumée (A2) fait tourner ce code en tout petit : regarder sa ligne « extension décision ».
- [ ] **A11. Niveau 2 (GPU, long).** D'abord chronométrer :
  `uv run scripts/evaluate_decision.py --all --level 2 --games 4 --game-steps 10 --no-save`, et lire l'estimation affichée par
  `smoke_test.py --time` (durée par run, réglages par défaut). Puis `uv run scripts/evaluate_decision.py --all --level 2`
  (6 runs : régression et diffusion du cœur). **Durée inconnue**, à ajouter aux ≈ 38 h du plan ; si elle est trop longue, réduire
  `--games` ou `--candidates` pour **tous** les runs, et le noter. Relancer ensuite `make analyze` (A7).

## B. Avec des humains (après l'analyse)

- [ ] **Relecture de 100 cas** pour estimer les erreurs du classifieur, et **test A/B en aveugle** (20 paires × 3 relecteurs) :
  `uv run scripts/human_review.py --help`. Il faut trouver trois relecteurs. Protocole et limites : `docs/EXPERIMENTS.md` §6.

## C. Décisions à prendre (avant A6)

- [x] **C1. Où est le « modèle du monde » ?** Tranché le 15 novembre : l'extension « décision » est implémentée (estimation d'un
  risque, puis planification dans le modèle). Restent hors du dépôt une récompense apprise et un agent entraîné dans le modèle
  (trop coûteux pour un T4).
- [ ] **C2. H2b sera probablement réfutée même avec un débruiteur parfait.** Sur le modèle jouet (p = 0,3), l'échantillonneur d'Euler à
  n = 3 donne p̂ ≈ 0,19 (erreur ≈ 0,11) à cause de la discrétisation, pas de l'entraînement (`docs/EXPERIMENTS.md` §5). Décider avant
  les runs : garder H2b telle quelle et la lire comme « n = 3 pas d'Euler ne suffit pas », ou changer d'échantillonneur / de n. Ce calcul
  est une idéalisation (1 dimension, débruiteur exact), pas une prédiction pour le réseau entraîné.
- [ ] **C3. Choix d'interprétation à confirmer ou à changer** (marqués **[choix]** dans `docs/EXPERIMENTS.md`, modifiables sans toucher
  au reste) :
  - les 3 contrôles = oracle (H0c), régression à p = 0 (H0d), diffusion à p = 0 ;
  - « grille » = grille de jetons 16 × 16 à K = 256, une graine ;
  - ρmax réglé sur du bruit synthétique σ = 0,01 (quantile 99,5 %, ρmax ≈ 0,16) ;
  - H0b évalué sur le rapport des MSE **en espérance** (un échantillonneur parfait a un rapport de 2, une régression idéale de 1) ;
  - qualité à l'horizon 16 = agent *et* obstacle suivis à ≤ 3 px ;
  - réanimation des codes morts du VQ activée par défaut ; pas de moyenne EMA par défaut ;
  - budget apparié (pas × lot), famille C répartie 40 % VQ / 60 % Transformer.
- [ ] **C4. Puissance statistique.** Avec 3 graines contre 3, la puissance n'est que de 0,46 pour un effet d = 2 (seuil non apparié
  2,27 s ; Bonferroni : 3,23 / 3,96 / 4,57 s pour 3 / 6 / 10 comparaisons). Accepter, ou prévoir plus de graines pour les runs du cœur.
- [ ] **C5. Budget et ordre de sacrifice.** Fixer le budget réel en heures de GPU et les groupes sacrifiés d'avance
  (`--skip-groups p_axis grid`, `--max-hours`).
- [ ] **C6. H5a sera probablement réfutée si les modèles sont bons.** Mesuré sur le vrai jeu avec des modèles factices : une image
  moyenne *exacte* lue par le lecteur doux donne 0,02 contact par partie, contre 0,16 pour le vrai jeu échantillonné 4 fois. H5a
  (« la régression subit au moins 0,5 contact de plus ») ne peut donc être confirmée que si la régression réelle se dégrade en
  recevant ses propres images floues. Décider avant A11 : garder H5a telle quelle (une réfutation est un résultat, comme pour H2b),
  ou la reformuler (par exemple avec le lecteur à seuil : 0,60 contact pour l'image moyenne exacte).
- [ ] **C7. Seuils et réglages de l'extension** (tous **[choix]**, `docs/EXTENSION_DECISION.md` §7) : 0,10 (H4a) ; 0,02 et 0,05 (H4b) ;
  0,5 contact (H5a) ; H0f (le critère « +3 points » passe de peu : le score du bot aléatoire varie de −2 à −3,3 selon les parties) ;
  J = 64, H = 8, m = 4, λ = 0,5, 50 parties, lecteur doux, p = 0,3 seulement ; 3 000 états de test au niveau 1.
- [ ] **C8. Récompense lue dans les pixels.** Elle est supposée connue (PlaNet l'apprend). C'est la limite principale du niveau 2 ;
  l'apprendre demanderait de ré-entraîner. À assumer dans le rapport, ou à lever.

## D. Non vérifié (à tester quand l'occasion se présente)

- [ ] **uv** : le vrai téléchargement PyPI et la suite de tests sont désormais vérifiés sous Windows 11 avec Python 3.12 (uv 0.11.2),
  et `uv.lock` est versionné. Reste la **vraie roue PyTorch** (CUDA ou CPU), macOS, et les autres versions de Python
  (`requires-python >= 3.10` est déclaré, vérifié seulement en 3.12/3.13). Si PyTorch n'a pas de roue pour votre Python :
  `uv sync --extra train --python 3.12`.
- [ ] **Gymnasium** : `uv sync --extra gym` puis `uv run pytest tests/test_env.py` ; l'enveloppe n'a jamais été exécutée avec le vrai
  Gymnasium (1 test ignoré).
- [ ] **Terminaux et encodage.** Mesuré le 15 novembre sous Windows 11 (Python 3.12) : Python y écrit sa sortie en `cp1252`, donc tout
  script qui affiche des accents, `δ`, `ρ` ou les cadres du jeu plante sur un `UnicodeEncodeError` (`play.py`,
  `validate_instrument.py`, `validate_decision.py`, `human_review.py`, `pytest`). Les rapports JSON sont écrits avant l'affichage,
  mais le code de sortie non nul fait croire à un échec de contrôle. Contournement vérifié : préfixer `PYTHONUTF8=1` (tout passe).
  **Correctif durable à décider** : forcer l'UTF-8 une fois pour toutes dans `scripts/_bootstrap.py`
  (`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`, idem `stderr`) — deux lignes, mais c'est du code.
  Reste à essayer : le clavier interactif (`msvcrt`) dans Windows Terminal, et Terminal.app sous macOS (`--colors 256`).
- [ ] **Extension « décision », côté PyTorch** : `evaluate_decision_run`, `plan_run`, le découpage des gros lots dans
  `TorchPredictor` (`max_batch = 512`, à baisser si la mémoire du GPU manque) et le débit d'inférence du test de fumée n'ont jamais
  tourné. Les jetons sont exclus du niveau 2 par défaut (coût) : `--families regression diffusion tokens` pour les inclure.
- [ ] **Extension « décision », robustesse** : une seule graine de parties (50 parties) pour les chiffres de référence ; le lecteur de
  contact suppose la palette par défaut (sur le rail, seul l'ennemi est rouge) : H0e le revérifie si vous changez les couleurs.
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
