# notebooks/

`flou_colab.ipynb` : déroule le projet sur Colab (GPU T4) — tests, test de fumée du code PyTorch, données, validation de
l'instrument, plan et budget, exécution reprenable, analyse. **Aucune cellule n'a été exécutée par l'auteur** : le code PyTorch n'a
jamais tourné là où il a été écrit ; la cellule « test de fumée » est obligatoire avant tout entraînement.

La logique réutilisable doit vivre dans `src/flou/` et `scripts/`, pas dans les notebooks.

**Pourquoi pip ici et pas uv.** Sur Colab, PyTorch (avec CUDA) est déjà installé ; le notebook n'installe que ce qui manque avec `pip`. Sur votre propre machine, utilisez uv (README, §2).
