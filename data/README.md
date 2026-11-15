# data/

Jeux de données générés par `scripts/generate_data.py` (ignorés par git, voir `.gitignore`).

```
data/p0.30/train_random.npz   train_greedy.npz     états, actions, branches, événements
data/p0.30/val_random.npz     val_greedy.npz
data/p0.30/test_bifurcation.npz                    1 000 états de bifurcation figés + contextes
```

Un dossier par valeur de p. Chaque fichier contient sa propre configuration de jeu (JSON).
Le format est décrit dans `src/flou/data/generate.py`.
