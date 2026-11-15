"""Instruments de mesure du projet (NumPy seulement : aucune dépendance à PyTorch).

classifier   classifieur contrefactuel (α̂, ρ), contrôle H0a, réglage de ρmax
stats        intervalles binomiaux, règles de décision à 3 graines, puissance, κ de Cohen
probe        sonde analytique de position (référence pour la sonde apprise)
recall       rappel du petit objet, objets hallucinés, artefacts (hors palette, blocs)
predictors   interface commune `Predictor` / `Batch`
baselines    image suivante = image courante
fakes        prédicteurs factices dont la bonne réponse est connue (bancs d'essai)
evaluate     évaluation sur le jeu de test de bifurcation, MSE contre plancher de Bayes, contrôles
rollout      rollouts de 16 pas : suivi de l'agent, rappel, dérive

Extension « décision » (docs/EXTENSION_DECISION.md ; guidée par PETS, PlaNet et Objective Mismatch) :
decision             niveau 1 : lecteur de contact, coût exact, erreur d'estimation du risque (H0e, H4)
planning             niveau 2 : planificateur par tirage de suites d'actions dans le modèle (H0f, H5)
decision_instrument  contrôles H0e et H0f, valeurs attendues « si les modèles étaient parfaits »
decision_analysis    verdicts H4 et H5, tables
"""
