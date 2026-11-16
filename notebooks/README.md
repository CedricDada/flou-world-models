# notebooks/

`flou_kaggle.ipynb` : **la voie retenue pour la campagne.** Déroule le projet sur Kaggle (GPU T4 x2) — état repris d'un
Dataset privé, dépôt cloné, données, validation de l'instrument, entraînement reprenable, contrôle du surapprentissage,
puis publication d'une nouvelle version du Dataset. Une session est un « Save & Run All (Commit) », qui tourne **en
arrière-plan** : la machine peut être éteinte. La variable `STAGE` de la première cellule dit ce que la session fait.
Raison du choix : `docs/JOURNAL.md`, entrée J8.

`flou_colab.ipynb` : le même déroulé sur Colab (GPU T4) — tests, test de fumée du code PyTorch, données, validation de
l'instrument, plan et budget, exécution reprenable, analyse. Il a servi au premier entraînement réel (incident J6) et
reste utile pour une vérification courte et interactive, mais **pas pour la campagne** : avec Colab Pro, le runtime est
libéré dès que la fenêtre se ferme ou que la machine se met en veille. Sa cellule qui archive le run à 100 000
transitions puis vide `runs/` est destructrice : ne jamais la relancer.

Dans les deux cas, la cellule « test de fumée » est obligatoire avant tout entraînement sur un environnement neuf.

**Interface web de Colab ou extension VS Code ?** Les deux exécutent le même runtime distant ; le dépôt doit être cloné dessus
dans les deux cas (le disque de votre machine n'est pas visible du noyau). Seul `drive.mount()` diffère : l'autorisation est
demandée à la page du notebook et peut échouer depuis VS Code (`MessageError: [dfs_ephemeral] Credentials propagation
unsuccessful`). La première cellule le détecte et affiche `persistant : True` ou `False` ; si c'est `False`, les résultats
vivent sur le disque de la session, les entraînements ne sont plus reprenables, et la campagne doit se lancer depuis le
navigateur. Si le montage réussit, l'extension VS Code convient pour tout.

La logique réutilisable doit vivre dans `src/flou/` et `scripts/`, pas dans les notebooks.

**Pourquoi pip ici et pas uv.** Sur Colab, PyTorch (avec CUDA) est déjà installé ; le notebook n'installe que ce qui manque avec `pip`. Sur votre propre machine, utilisez uv (README, §2).
