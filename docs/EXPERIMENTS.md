# Plan d'expériences, règles de décision et état de vérification

Ce document dit **quoi lancer, dans quel ordre, comment lire les verdicts, et ce qui a été vérifié ou non**. Il complète le
cours (`docs/cours/`, chap. 7 et 9) et les descriptions de projet (`docs/projet/`) ; en cas d'écart avec le dossier
initial, le cours fait foi. Les choix d'interprétation que j'ai dû faire sont marqués **[choix]** ; ils ont tous été tranchés le 15 novembre 2026, avant tout
entraînement, dans [`DECISIONS.md`](DECISIONS.md). Ils sont modifiables sans
toucher au reste (`src/flou/train/runs.py` pour le plan, `src/flou/eval/analysis.py` pour les règles).

## 1. Ordre d'exécution

Les commandes supposent l'environnement installé par `uv sync` (et `uv sync --extra train` sur la machine GPU) : voir le README, §2. Sans uv, remplacer `uv run scripts/x.py` par `python scripts/x.py`.

| Étape | Commande | Machine | Ce qui en sort |
|---|---|---|---|
| 0 | `make test` | tout | tests NumPy (jeu, bots, instruments, analyse, outils humains) |
| 1 | `uv run scripts/smoke_test.py --time` | **GPU** | vérifie le code PyTorch de bout en bout ; écrit `reports/timing.json` |
| 2 | `uv run scripts/generate_data.py …` | tout | `data/p*/` : p = 0 ; 0,1 ; 0,3 ; 0,5, train, val **et test** (voir `make data`) |
| 3 | `uv run scripts/validate_instrument.py` | tout | `reports/instrument.json` : ρmax, contrôle H0a |
| 4 | `uv run scripts/plan.py --timing reports/timing.json --budget-hours 38` | tout | nombre de pas par run qui tient dans le budget |
| 5 | `uv run scripts/run_plan.py --steps N` | **GPU** | `runs/<run>/` : entraînement puis évaluation, reprenable |
| 6 | `uv run scripts/analyze.py` | tout | `reports/` : verdicts, tables, figures, rapport |
| 7 | `uv run scripts/human_review.py …` | tout + **humains** | relecture de 100 cas, test A/B en aveugle |
| 8 | `uv run scripts/validate_decision.py` | tout (≈ 5 min) | extension « décision » : `reports/decision_instrument.json`, contrôles H0e et H0f |
| 9 | `uv run scripts/evaluate_decision.py --all` | **GPU** | niveau 1 (H4) : `runs/<run>/decision.json`, runs du cœur |
| 10 | `uv run scripts/evaluate_decision.py --all --level 2` | **GPU**, long | niveau 2 (H5) : `runs/<run>/planning.json`, régression et diffusion du cœur |
| 11 | `uv run scripts/analyze.py` | tout | le rapport contient alors H4 et H5 |

Les étapes 8 à 11 forment l'extension « décision » ; elles sont facultatives et ne changent rien aux étapes 0 à 7. Hypothèses,
définitions et articles qui les guident (PETS, PlaNet, Objective Mismatch) : [`EXTENSION_DECISION.md`](EXTENSION_DECISION.md).

Répétition générale sans GPU : `uv run scripts/dry_run.py` (runs **simulés**, tout est marqué SIMULATION, écrit dans
`runs_dryrun/` et `reports/dryrun/`, jamais dans `runs/` ni `reports/`).

## 2. Les 25 entraînements

Ordre de priorité décroissante (c'est l'ordre d'exécution ; si le calcul manque, on sacrifie par la fin).

| Groupe | Runs | Contenu | Rôle |
|---|---|---|---|
| cœur | 9 | A, B, C × 3 graines, p = 0,3 (C : K = 256, grille 8 × 8) | H1, H2, H3 |
| contrôles | 3 | oracle p = 0,3 ; régression p = 0 ; diffusion p = 0 | H0c, H0d, contrôle de n = 1 |
| axe K | 6 | C, K ∈ {64, 1 024} × 3 graines | H3a, H3b |
| grille | 1 | C, K = 256, grille 16 × 16, 1 graine | sépare l'effet de la grille de celui de K |
| axe p | 6 | A, B, C × p ∈ {0,1 ; 0,5}, 1 graine | exploratoire (flou en fonction de p) |

**Budget apparié.** Même nombre de pas × même taille de lot (donc mêmes images vues) pour toutes les familles ; pour la famille
C, le VQ-VAE et le Transformer se partagent ce total (40 % / 60 %). Pas d'EMA par défaut, pour que la comparaison ne dépende pas
d'un réglage propre à une famille. Les paramètres sont proches mais pas égaux : 5,87 M (A et oracle), 6,01 M (B), 6,04 M (C,
K = 256). **Attention à l'axe K** : K = 64 pèse 5,91 M, K = 1 024 pèse 6,59 M (+ 9 % par rapport à K = 256, + 12 % par rapport à A) à
cause du dictionnaire et de la table d'embeddings ; si H3a est confirmée, ce n'est pas seulement K qui varie.

**[choix] Les trois contrôles.** Le cours fixe leur nombre (3) et les hypothèses H0c, H0d et « la perte à n = 1 n'est pas propre à
la stochasticité », pas la liste. Lecture retenue : régresseur *oracle* (reçoit la branche tirée) ; régression à p = 0 ;
diffusion à p = 0 évaluée à n = 1.

**[choix] La « grille ».** Le plan compte « grille : 1 » sans la définir. Lecture retenue : une grille de jetons plus fine
(16 × 16, quatre fois plus de jetons) à K fixé, une seule graine.

**Durée.** Le cours annonce ≈ 1,5 h par run (≈ 38 h au total). Cette durée n'a **jamais été mesurée** : le défaut de 20 000 pas
× lot 64 est une valeur de départ. `smoke_test.py --time` mesure le temps par pas sur votre GPU, `plan.py --budget-hours`
en déduit le nombre de pas.

## 3. L'instrument de mesure

*Établi (calculé et testé ici).* Chaque prédiction ŷ est projetée sur le segment [B, A] : α̂ = ⟨ŷ − B, A − B⟩ / δ², bornée à
[0, 1], et ρ = distance au segment / δ. Avec ε = 0,2 : **nette-A** si α̂ ≥ 0,8, **nette-B** si α̂ ≤ 0,2, **floue** entre les deux,
**autre** si ρ > ρmax (objet manquant, déplacé, artefacts). Le mélange idéal a α̂ = p : il n'est donc « flou » que pour
0,2 < p < 0,8 ; à p = 0,1 une régression parfaite est classée nette-B. Un petit objet détruit donne ρ ≈ 0,325 sans bruit (≈ 0,35 mesuré avec σ = 0,01), d'où ρmax < 0,32, c'est-à-dire un
bruit de génération σ ≲ 0,015 pour que le classifieur sépare encore une image bruitée d'une image cassée.

- **[choix] ρmax** est réglé sur des images pures bruitées par un bruit **synthétique** (σ = 0,01, quantile 99,5 %) : ρmax ≈ 0,16.
  C'est un choix de l'instrument, pas une propriété d'un modèle. `instrument.json` donne la sensibilité à × 0,5 / × 1 / × 2 :
  à × 0,5 plus aucune image pure n'est reconnue ; à × 2 (≈ 0,32) la détection du petit objet détruit passe encore avec σ = 0,01
mais n'aurait plus de marge pour un bruit plus fort. **La part de
  « autre » dépend donc de ce choix ; ne jamais la lire comme une mesure absolue.**
- **H0a est un test statistique** : ρmax est réglé pour classer 99,5 % des images pures, le contrôle exige plus de 99 %. Avec
  peu d'images la marge de 0,5 point ne tient pas : mesuré ici, le contrôle échoue 14 fois sur 40 tirages à 120 paires,
  0 fois sur 30 à 450 paires, et réussit à 3 000 paires (1 000 états × 3 valeurs de p, 99,47 % d'images pures bien classées). Avec les quatre valeurs de
  p du plan (0 ; 0,1 ; 0,3 ; 0,5) on dispose de 4 000 paires.
  `validate_instrument.py` avertit sous 600 paires. Un échec de H0a sur un petit jeu ne prouve pas que l'instrument est faux.
- **MSE.** Le plancher de Bayes est p(1−p)δ² (somme des carrés, échelle 0–1). Un régresseur idéal a un rapport MSE/plancher égal à
  1,00 *en espérance* ; un échantillonneur parfait a un rapport 2 (il tire une branche au hasard). La MSE ne distingue donc pas
  « net et bien calibré » de « flou » dans le sens attendu : d'où le classifieur. Le rapport est calculé *en espérance* (contre
  la loi exacte des deux futurs) et *tiré* (contre le futur réellement advenu) ; un oracle qui fuit l'information a un rapport
  tiré proche de 0. **[choix]** H0b s'évalue sur le rapport en espérance.
- **Sonde de position.** La sonde de référence des rollouts est analytique (sans apprentissage). La sonde CNN
  (`train_probe.py`) n'est qu'un recoupement, utilisable seulement si son erreur de validation est < 0,5 px.
  **[choix]** La qualité à l'horizon 16 est la part de rollouts où l'agent *et* l'obstacle sont suivis à ≤ 3 px. La
  classification par sonde (< 2 px) du dossier initial est remplacée par le classifieur contrefactuel du cours.
- **Mémoire.** La validation de l'instrument travaille par blocs de 250 paires : à 3 000 paires elle tient dans ≈ 0,8 Go
  (une première version non découpée dépassait 8 Go et était tuée).

## 4. Règles de décision

Appliquées par `flou.eval.analysis.verdicts` ; les statuts sont *confirmée / réfutée / non concluante / non évaluable* pour une
hypothèse, *réussi / échoué / non évaluable* pour un contrôle. Un résultat sous le seuil à 3 graines est « non concluant », jamais
« pas d'effet » : les points bruts sont dans `verdicts.json`.

| | Règle |
|---|---|
| H0a | erreur moyenne sur α̂ < 0,05 et plus de 99 % des images pures bien classées (sans entraînement) |
| H0b | rapport MSE/plancher (en espérance) dans [1,00 ; 1,10] à chaque graine et pas de fuite d'information |
| H0c, H0d, n = 1 à p = 0 | moins de 5 % de flou |
| H1 | **confirmée** si flou(régression) > 50 % et nettes(diffusion n = 3) > 90 % ; **réfutée** si flou(régression) < 30 % ou flou(diffusion n = 3) > 10 % |
| H2a | comparaison *appariée* n = 1 contre n = 3 (mêmes débruiteurs) : **confirmée** si écart ≥ 10 points et > 2,48 s_d ; **réfutée** si < 10 points |
| H2b | **confirmée** si \|p̂ − p\| < 0,05 à chaque graine avec ≥ 1 000 images nettes ; **réfutée** si ≥ 0,05 à chaque graine |
| H3a | comparaison non appariée K = 64 contre K = 1 024 : **confirmée** si rappel(64) < rappel(1 024) de plus de 2,27 écarts-types poolés ; **réfutée** si l'écart n'a pas le bon signe |
| H3b | exploratoire : corrélations de rang rappel / MSE-pixel contre qualité à l'horizon 16, sans seuil |
| H0e | le lecteur de contact retrouve l'événement du jeu sur ≥ 99 % des vraies images (sans entraînement) |
| H0f | planificateur avec le vrai jeu comme modèle : ≤ 0,5 contact par partie et ≥ 3 points de plus que le bot aléatoire (sans entraînement) **[choix]** |
| H4a | lecteur à seuil, k = 1, non appariée : **confirmée** si E(régression) − E(diffusion n = 3) ≥ 0,10 et > 2,27 écarts-types poolés ; **réfutée** si < 0,10 **[choix]** |
| H4b | lecteur doux, k = 1 : **confirmée** si E(régression) ≤ E(diffusion n = 3) + 0,02 ; **réfutée** si > + 0,05 **[choix]** |
| H4c | exploratoire : erreur selon l'horizon k, sans seuil |
| H5a | contacts par partie, non appariée : **confirmée** si régression − diffusion n = 3 ≥ 0,5 et > 2,27 écarts-types poolés ; **réfutée** si < 0,5 **[choix]** |
| H5b | exploratoire : corrélations de rang entre le score du planificateur et la part d'images nettes, la MSE par pixel, le suivi à 16 pas |

H0e à H5b sont appliquées par `flou.eval.decision_analysis.verdicts` ; leur origine (PETS, arXiv:1805.12114 ; PlaNet,
arXiv:1811.04551 ; Objective Mismatch, arXiv:2002.04523) est donnée ligne par ligne dans `EXTENSION_DECISION.md`.

Les seuils (2,27 s pour 3 contre 3 ; 2,48 s_d apparié) sont des **règles de décision à très peu de graines**, pas des preuves.
Avec k comparaisons simultanées il faut les relever (Bonferroni, k = 3 / 6 / 10 : 3,23 / 3,96 / 4,57 s non apparié) ; avec
3 graines la puissance pour un effet de 2 écarts-types n'est que d'environ 46 %. Un H3b sur 9 points (3 K × 3 graines) est très
bruité.

## 5. Ce qu'on peut attendre — et ce qu'on ne sait pas

*Calculé sur un modèle jouet* (débruiteur exact d'un mélange de deux points, reproduit par `flou.models.edm_math`) : avec σmax = 80
sur l'échelle [−1, 1] (soit ≈ 5,3 δ ici) et p = 0,3, l'échantillonneur d'Euler donne p̂ ≈ 0,19 à n = 3, 0,23 à n = 5, 0,27 à n = 20,
0,28 à n = 50 ; avec σmax = 80 δ, p̂ ≈ 0,015 à n = 3. Autrement dit **un débruiteur parfait échouerait probablement à H2b à n = 3**
(erreur ≈ 0,11) à cause de la discrétisation d'Euler, pas à cause de l'entraînement. Ce jouet est une idéalisation
(1 dimension, débruiteur exact) : c'est une raison de s'attendre à une H2b réfutée, pas une prédiction pour le U-Net entraîné. Si
elle l'est, la lire comme « n = 3 pas d'Euler ne suffit pas à retrouver p », et regarder la courbe p̂(n)
(`calibration_par_n.csv`, `fig_calibration.png`) avant de conclure sur la diffusion en général.

*Intuition (non vérifiée).* La régression sera floue pour 0,2 < p < 0,8 ; la diffusion à n = 1 se comporte comme la régression ;
la famille C perdra le petit objet à petit K. Rien de cela n'est mesuré ici : c'est ce que les runs doivent établir.

## 6. Relecture humaine

| Outil | Commande | Ce qu'il produit |
|---|---|---|
| 100 cas stratifiés | `human_review.py review-make --select <run>[:variante] …` | planches [prédiction \| A \| B] sans la classe automatique, `review.csv` à remplir |
| score | `human_review.py review-score --dir …` | matrice de confusion, accord par classe (Clopper-Pearson), κ, exactitude pondérée par la prévalence |
| A/B en aveugle | `human_review.py ab-make --a … --b …` | 20 paires, mêmes contextes, ordre gauche/droite aléatoire, 3 fichiers relecteur, `key.csv` séparée |
| score A/B | `human_review.py ab-score --dir … --responses …` | préférence + intervalle, κ moyen entre relecteurs, part de « L » par relecteur |

Limites à garder en tête : l'échantillon de relecture est stratifié (le taux d'erreur global se lit par l'exactitude *pondérée*, pas
par l'accord brut) ; le réservoir est plafonné à 25 images par (variante, classe), ce qui sur-représente légèrement les petites
classes ; 20 paires donnent un intervalle large (15 préférences sur 20 → [0,51 ; 0,91]) ; l'ordre gauche/droite est le même pour les
trois relecteurs, donc un biais de position apparaît comme de l'accord (vérifier la part de « L »).

## 7. Correspondance avec le code

| Sujet | Module |
|---|---|
| plan, budget, estimation de durée | `src/flou/train/runs.py`, `scripts/plan.py` |
| modèles (plans de paramètres, EDM, VQ-VAE, GPT) | `src/flou/models/` |
| entraînement reprenable, tâches par famille, évaluation | `src/flou/train/` |
| classifieur, statistiques, rappel, sonde, rollouts | `src/flou/eval/` |
| verdicts, tables, figures, rapport | `src/flou/eval/analysis*.py`, `report.py`, `scripts/analyze.py` |
| relecture humaine, A/B | `src/flou/eval/human.py`, `scripts/human_review.py` |
| répétition générale | `src/flou/eval/dryrun.py`, `scripts/dry_run.py` |
| extension « décision » : lecteur de contact, coût exact, planificateur, contrôles, verdicts | `src/flou/eval/decision.py`, `planning.py`, `decision_instrument.py`, `decision_analysis.py`, `scripts/validate_decision.py` |
| extension « décision » : mesure des runs entraînés | `src/flou/train/evaluation.py` (`evaluate_decision_run`, `plan_run`), `scripts/evaluate_decision.py` |

## 8. État de vérification

| | Statut |
|---|---|
| Jeu, bots, données, classifieur, statistiques, rappel, rollouts, analyse, rapport, relecture humaine, scripts sans PyTorch | **exécutés et testés** (`make test`) |
| Chaîne d'analyse sur 25 runs simulés | **exécutée** : les effets plantés sont retrouvés ; aucune valeur scientifique |
| Modèles PyTorch, boucle d'entraînement, reprise, évaluation, `FrameBank`, sonde CNN, `train.py`, `evaluate.py`, `run_plan.py`, `smoke_test.py` | **écrits, jamais exécutés** (pas de PyTorch dans l'environnement de rédaction) ; seuls les noms sont vérifiés par `scripts/lint_names.py` et les plans de paramètres par des tests NumPy |
| Extension « décision », partie NumPy : lecteur de contact, coût exact, planificateur, H0e et H0f sur le vrai jeu, verdicts H4/H5 sur runs simulés | **exécutée et testée** (49 tests) ; H0e : 100 % d'accord ; H0f : 0,16 contact par partie, +3,16 points sur le bot aléatoire (seuil 3 : passe de peu) |
| Extension « décision », mesure des modèles : `evaluate_decision_run`, `plan_run`, `evaluate_decision.py`, découpage des lots dans `TorchPredictor` | **écrits, jamais exécutés** ; le test de fumée les fait tourner en tout petit |
| Entraînement réel, temps par run, résultats H1–H5, relectures humaines, rapport final | **non faits** : ils demandent un GPU et des relecteurs |
| Enveloppe Gymnasium, terminaux Windows / macOS | non testés (voir README) |
| uv : `uv lock`, `uv sync`, extras, `uv run`, `Makefile` | testés **hors ligne** sur un index de roues local ; **non testés** : téléchargement réel depuis PyPI, vraie PyTorch, Windows / macOS, Python autre que 3.13 ; `uv.lock` non fourni |

Premier réflexe sur un GPU (après `uv sync --extra train`) : `uv run scripts/smoke_test.py --time`. S'il échoue, la trace dit quel fichier corriger ; la plupart des
erreurs probables sont de simples écarts de forme de tenseurs ou d'API PyTorch qu'une exécution réelle révèle en quelques minutes.
