# notebooks/

`flou_colab.ipynb` : déroule le projet sur Colab (GPU T4) — tests, test de fumée du code PyTorch, données, validation de
l'instrument, plan et budget, exécution reprenable, analyse. **Aucune cellule n'a été exécutée par l'auteur** : le code PyTorch n'a
jamais tourné là où il a été écrit ; la cellule « test de fumée » est obligatoire avant tout entraînement.

**Interface web de Colab ou extension VS Code ?** Les deux exécutent le même runtime distant ; le dépôt doit être cloné dessus
dans les deux cas (le disque de votre machine n'est pas visible du noyau). Seul `drive.mount()` diffère : l'autorisation est
demandée à la page du notebook et peut échouer depuis VS Code (`MessageError: [dfs_ephemeral] Credentials propagation
unsuccessful`). La première cellule le détecte et affiche `persistant : True` ou `False` ; si c'est `False`, les résultats
vivent sur le disque de la session, les entraînements ne sont plus reprenables, et la campagne doit se lancer depuis le
navigateur. Si le montage réussit, l'extension VS Code convient pour tout.

La logique réutilisable doit vivre dans `src/flou/` et `scripts/`, pas dans les notebooks.

**Pourquoi pip ici et pas uv.** Sur Colab, PyTorch (avec CUDA) est déjà installé ; le notebook n'installe que ce qui manque avec `pip`. Sur votre propre machine, utilisez uv (README, §2).
