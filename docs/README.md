# Documentation

| Fichier | Contenu | Quand le lire |
|---|---|---|
| [`cours/cours_world_models_flou.pdf`](cours/cours_world_models_flou.pdf) (+ `.tex`) | Cours préparatoire de 21 pages : régression et moyenne conditionnelle, diffusion EDM, jetons discrets, mesure (calibration, graines, tests), ingénierie, et l'extension « décision » (§9.5). Les extraits de code PyTorch du cours sont la base de `src/flou/models/`. | Avant d'entraîner quoi que ce soit. Chap. 2 et 7 pour les instruments, chap. 9 pour le plan de 25 entraînements. |
| [`projet/projet_world_models_flou_final.pdf`](projet/projet_world_models_flou_final.pdf) (+ `.tex`) | Énoncé de 34 pages destiné aux étudiants : question, hypothèses (H0 à H5), protocole, évaluation, livrables, prompt tutoral. | Pour savoir ce qui est demandé et comment ce sera évalué. |
| [`projet/projet_world_models_flou_2pages.pdf`](projet/projet_world_models_flou_2pages.pdf) (+ `.tex`) | Version courte : question, hypothèses et seuils, instrument et contrôles, extension « décision » en un paragraphe. | Pour une lecture de cinq minutes. |
| [`projet/projet_world_models_flou_dossier_initial.md`](projet/projet_world_models_flou_dossier_initial.md) | Dossier d'origine (Markdown). Compte 19 entraînements (≈ 28 h) ; le cours en compte 25 (≈ 38 h) et fait foi. | Pour l'historique et la liste des sources. |
| [`DECISIONS.md`](DECISIONS.md) | Les décisions tranchées **avant le premier entraînement** (bloc C du `TODO.md`) : pour chacune, l'hypothèse concernée, la décision et ses raisons, plus la règle de révision. | Avant de lancer le premier run, et pour comprendre pourquoi un seuil n'a pas été déplacé. |
| [`GAME_SPEC.md`](GAME_SPEC.md) | Spécification exacte du mini-jeu (dynamique, rendu, bifurcation, constantes). | Pour modifier le jeu ou écrire un autre instrument. |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Plan des 25 entraînements, définition des contrôles, règles de décision et correspondance avec les modules du code. | Avant de lancer `scripts/run_plan.py`. |
| [`EXTENSION_DECISION.md`](EXTENSION_DECISION.md) | Extension « décision » : le flou fausse-t-il un risque estimé (H4) et coûte-t-il des points à un agent qui planifie dans le modèle (H5) ? Définitions, seuils, articles qui guident chaque choix, mesures faites sans modèle. | Avant `scripts/evaluate_decision.py`. |
| `img/` | Figures du README (images contrefactuelles, parties par bot). | |

Les figures du cours se régénèrent avec `cours/fig/toy.py`, `euler_toy.py` et `euler_fig.py` (NumPy et matplotlib seulement) ;
`cours/check_numbers.py` recalcule les nombres cités dans le texte (seuils 2,27 s et 2,48 s, tailles d'échantillon, puissance, et pour l'extension « décision » l'erreur 0,30 du lecteur à seuil et l'erreur 0,133 d'un générateur calibré à 8 tirages).

**Précédence en cas de désaccord** : le cours (chap. 9) l'emporte sur le dossier initial. Les valeurs que le cours qualifie de
« proposées » (distance minimale entre branches de 8 pixels, au moins 5 % de transitions de bifurcation) sont des paramètres de
`GameConfig`, pas des constantes cachées.
