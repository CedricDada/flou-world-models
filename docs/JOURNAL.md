# Journal de bord : problèmes rencontrés, solutions testées

Ce fichier garde la **trace du chemin**, pas seulement son résultat. Un dépôt qui ne montre que l'état final laisse croire que
tout est tombé juste du premier coup ; or ce qui s'est passé entre les deux est souvent le plus instructif — et, pour un
protocole pré-enregistré, c'est aussi ce qui permet de distinguer une correction légitime d'une hypothèse ajustée après coup.

Chaque entrée suit le même ordre : **symptôme observé**, **cause établie**, **solution retenue**, **statut**, et **où se trouve
la preuve**. Le statut dit exactement ce qui a été vérifié :

| Statut | Ce que ça veut dire |
|---|---|
| **Résolu** | Correctif appliqué et vérifié par une exécution |
| **Contourné** | Le problème subsiste, on sait l'éviter ; le correctif de fond reste à décider |
| **En cours de vérification** | Cause établie, correctif choisi, mesure de contrôle pas encore faite |
| **Décidé, non exécuté** | Décision prise et tracée, l'étape concernée n'a pas encore tourné |

Les décisions de protocole, elles, vivent dans [`DECISIONS.md`](DECISIONS.md) : ce journal y renvoie, il ne les remplace pas.

---

## J7 — Relecture externe : deux défauts d'analyse trouvés avant la campagne

**16 novembre 2026.** Les dossiers `docs/cours/` et `docs/projet/` ont été confiés à un agent externe pour relecture
bibliographique. Il a rendu une version 2.2 des trois `.tex` et un correctif de code. Deux de ses trois points étaient des
défauts réels, invisibles tant qu'aucun run n'avait tourné.

**Défaut 1 : H2b ne pouvait pas être concluante.** `verdicts()` exige `min_n_h2b = 1000` images **nettes** par graine, mais le
jeu de test ne comptait que 1 000 états avec une prédiction par état. Le plafond atteignable était donc exactement 1 000 : une
seule image floue ou « autre » suffisait à rendre le verdict « non concluant ». Le run archivé affichait 2,0 % d'« autre »,
donc c'était perdu d'avance. **Solution appliquée :** jeu de test porté à **2 000 états** par valeur de p. L'autre piste,
augmenter `--n-samples`, a été écartée : elle aurait dupliqué les prédictions déterministes de la régression et rétréci
artificiellement ses intervalles de confiance.

**Défaut 2 : la décision C4 n'était pas câblée.** `DECISIONS.md` fixait k = 3 comparaisons simultanées, mais `analysis.py`
appelait `compare_groups` sans `k_tests`, donc avec les seuils k = 1. **Solution appliquée :** `K_CORE = 3` passé à H2a et H3a.
Seuils vérifiés par le calcul : apparié n = 3 → **4,416** ; non apparié 3 v 3 → **3,234**.

**Réserve écrite, décision de ne pas en tenir compte.** Bonferroni corrige le nombre de *tests de signification* simultanés.
H1 n'en est pas un : c'est une règle à seuils fixes sur des proportions, sans test t ni valeur p. La famille ne contient donc
que H2a et H3a, soit k = 2 → 3,583 en apparié. Le choix de garder **k = 3** est conservateur et assumé : C4 l'a fixé avant les
runs, et desserrer un seuil pré-enregistré est précisément ce que le protocole interdit. À signaler dans le rapport.

**Conséquence que la relecture n'avait pas vue.** ρmax est réglé sur les jeux de test (4 000 paires sur 4 jeux) : passer à
2 000 états en fait 8 000, donc un quantile 99,5 % différent. Il faut relancer `validate_instrument.py`, revérifier H0a, et
évaluer tous les runs avec le même ρmax. Aucun entraînement n'est perdu pour autant : le jeu de test ne sert qu'à l'instrument
et à l'évaluation.

**Troisième point, non vérifiable par lecture :** H0f relancé aux réglages réduits (32 suites, 20 parties). La relecture
annonce 0,30 contact par partie pour un maximum de 0,5 et +4,90 points, donc réussi. La cellule du notebook lance désormais
`validate_decision.py --games 20 --candidates 32`. **Reste à confirmer par une exécution** (CPU seul).

**Correctif mineur accepté :** la docstring de `evaluate.py` affirmait qu'un régresseur parfait passe sous le plancher « une
fois sur trois ». C'est une fois sur deux : l'espérance de l'erreur contre l'image tirée vaut p(1−p)²δ² + (1−p)p²δ² = p(1−p)δ²,
soit exactement le plancher, donc l'estimateur est non biaisé.

**Outil ajouté à la demande de la relecture.** Sa réserve est juste : 400 000 transitions ne *garantissent* pas l'absence de
mémorisation, et H0b ne surveille que la régression — la diffusion et les jetons peuvent mémoriser sans qu'aucun contrôle ne
le voie. D'où [`scripts/training_curves.py`](../scripts/training_curves.py), qui lit les `log.jsonl`, écrit une figure par run
et lève deux alertes : perte de validation remontée de plus de 10 % depuis son minimum (valable pour les quatre familles), et
perte d'entraînement passée sous le plancher de Bayes (familles `regression` et `oracle` seulement, les seules dont la perte
soit une MSE ; le plancher est recalculé sur les données, pas codé en dur). Le script rend un code de sortie non nul en cas
d'alerte. Vérifié sur le run archivé de J6 : il lève bien les deux.

**Statut : appliqué**, sauf la confirmation de H0f, en attente d'exécution. Les `.tex` v2.2 annoncent encore 100 000
transitions d'entraînement ; une v2.3 a été demandée.

---

## J6 — La régression mémorise la branche au lieu de produire le flou

**15 novembre 2026, après le premier entraînement réel.** C'est l'incident le plus important du projet à ce jour : il touche
l'hypothèse centrale, et il a été détecté par un contrôle pré-enregistré, ce qui est précisément ce qu'on lui demandait.

**Symptôme.** Premier run du plan, `core-regression-p0.30-s0`, 34 800 pas, 2,69 h sur Tesla T4. L'évaluation donne :

```
floue 22,5 %   autre 2,0 %   p̂ = 0,158   MSE/plancher 1,47 (tirée 1,53)   rappel 99,0 %
```

Deux anomalies. Le contrôle **H0b** exige un rapport MSE/plancher dans [1,00 ; 1,10] : on est à **1,47**, donc `pass: false`.
Et la part de flou est de **22,5 %** là où la théorie en prédit la quasi-totalité : sous perte quadratique, le meilleur
prédicteur d'un futur à deux issues est le mélange p·A + (1−p)·B.

**Cause établie.** Le fichier `log.jsonl` du run donne la réponse sans ambiguïté :

| pas | perte d'entraînement | perte de validation |
|---|---|---|
| 2 000 | 0,001264 | 0,000917 |
| 6 000 | 0,000834 | 0,000878 |
| 14 000 | 0,000911 | 0,000803 |
| 20 000 | 0,000729 | 0,000924 |
| 26 000 | 0,000517 | 0,001069 |
| 30 000 | 0,000257 | 0,001164 |
| **34 800** | **0,000305** | **0,001195** |

La validation forme un plateau bruité entre les pas 4 000 et 18 000 — son minimum, 0,000800, tombe au pas 8 000, et vaut
0,000803 au pas 14 000, écart sans signification — puis monte de façon **monotone à partir du pas 20 000** pour finir à
0,001195, soit **+49 %** au-dessus du minimum, pendant que l'entraînement continue de descendre : courbe de surapprentissage
caractéristique. En nombre de passages sur les données, le plateau s'étend de 2,6 à 11,5 et la hausse soutenue commence à
**12,8**. Le chiffre qui tranche est ailleurs. Le **plancher de Bayes** de ce jeu de données,
calculé sur les données elles-mêmes (part de transitions de bifurcation 16,7 % ; δ² moyen 57,9 ; p(1−p) = 0,21), vaut
**0,000659** dans les unités de `mse_loss`. La perte d'entraînement finit à **0,000305, soit 0,46 × ce plancher**.

Aucun modèle qui ignore la branche ne peut passer sous ce plancher — c'est une borne mathématique, pas une difficulté
d'optimisation. Il a donc fallu vérifier qu'il l'ignore bien : dans [`regression.py`](../src/flou/models/regression.py), la
branche n'est injectée dans le réseau que si `oracle=True`, ce qui ne concerne que le run de contrôle `control-oracle`. Pas de
fuite non plus au test : `leak_z = −8,96`, négatif, donc la MSE contre l'image tirée est *au-dessus* de l'espérance.

La seule explication restante est la **mémorisation du jeu d'entraînement**. Le modèle a retenu quelle branche avait été tirée
dans chaque transition vue : 16 700 bifurcations à mémoriser, 5,9 M de paramètres, et **22 passages** sur les données
(2 227 200 images vues pour 100 000 transitions distinctes). Le contexte de 4 images identifie presque sûrement le couple
(épisode, instant) ; il suffit d'y associer l'issue.

Au test, sur des états jamais vus, cette mémoire ne sert à rien. Le modèle a appris à produire une image *nette* — puisque
c'était la bonne réponse sur l'entraînement — alors il en produit une. `eval.json` le montre finement : `alpha_mean = 0,241`,
pas si loin de p = 0,30, mais avec un écart-type d'environ 0,29 déduit du rapport MSE. Le modèle ne produit pas un mélange
stable : il propose un mélange **différent et erratique pour chaque état**, comme s'il tentait de deviner l'issue. Avec
`eps = 0,2`, 636 états passent sous le seuil et deviennent « nette-B », 119 au-dessus de 0,8 et deviennent « nette-A ».

À noter, parce que cela délimite le problème : les images sont par ailleurs excellentes. Rappel du petit objet 99,0 %, aucune
hallucination, rollouts à 16 pas stables (`quality_last = 0,98`). Ce n'est pas un défaut de rendu, c'est un défaut de
comportement **face à l'incertitude** — exactement la grandeur que le projet mesure.

**Solution retenue.** Le budget apparié fixe les images **vues** (2 227 200) ; il ne dit rien des images **distinctes**. Le
rapport des deux valait 22. Le jeu d'entraînement passe de **100 000 à 400 000 transitions** pour les quatre valeurs de p, ce
qui ramène ce rapport à **5,6**, soit le milieu du plateau sain observé au premier run. Rien d'autre ne bouge :
mêmes pas, même lot, même taux d'apprentissage, mêmes seuils, mêmes hypothèses.

Trois points de vérification ont précédé le choix. Les jeux train, val et test utilisent des **flux aléatoires indépendants**
([`generate.py`](../src/flou/data/generate.py), `stream_seeds`), donc quadrupler l'entraînement ne contamine ni la validation ni
le test, qui restent inchangés. La mémoire tient : `FrameBank` pré-rend toutes les images sur le GPU, environ 1,24 Go par
100 000 transitions, soit 4,96 Go de banque + 0,25 de validation + 3,7 de pic d'entraînement ≈ **8,9 Go** sur les 15,6 Go d'un
T4. Et le plafond matériel est connu : supprimer toute répétition demanderait 2,23 M de transitions, soit 27 Go, hors de portée
sans réécrire `FrameBank` pour rendre les images à la volée.

**Ce qu'il faut assumer.** Ce changement rend H1 (« régression floue, diffusion nette ») **plus facile** à confirmer, puisqu'il
rétablit le flou attendu de la régression. C'est le sens inverse de la règle 2 de [`DECISIONS.md`](DECISIONS.md), qui demande de
préférer l'option la plus sévère. Il n'est légitime que parce qu'il relève de la règle 3 : *on ne change un réglage que s'il
mesure autre chose que ce qu'on croit*. À 100 000 transitions, le dispositif mesurait la mémorisation d'un jeu de données, pas
le flou inhérent à la perte quadratique. La révision est tracée à sa date dans [`DECISIONS.md`](DECISIONS.md), et le rapport
final doit porter la mention de ce changement.

**Statut : en cours de vérification.** Avant d'engager les 28 h de campagne, un seul run est relancé aux nouvelles données
(2,7 h) et doit satisfaire deux critères : la validation ne remonte plus jusqu'au pas 34 800, et `ratio_expected` revient dans
[1,00 ; 1,10]. Le premier critère est désormais automatisé — `scripts/training_curves.py` trace les courbes et lève une alerte,
avec un code de sortie non nul (voir J7). Réserve honnête : le meilleur point de validation atteint valait 1,22 × le plancher, sur une grandeur différente
(toutes transitions, et non les seules bifurcations), donc non directement comparable — rien ne garantit l'atterrissage dans la
bande.

**Preuves conservées.** Le run fautif n'est pas supprimé. Ses deux fichiers de mesure sont versionnés dans le dépôt, sous
[`incidents/J6_surapprentissage/`](incidents/J6_surapprentissage/) : `log.jsonl` (348 points d'entraînement, 18 de validation)
et `eval.json` (verdicts, comptages, rollouts). Le dossier complet du run, checkpoints compris, est archivé hors dépôt sous
`archive/core-regression-p0.30-s0_N100k/`, les poids étant trop lourds pour Git.

La courbe de validation en U et une perte d'entraînement sous le plancher de Bayes constituent une illustration de
surapprentissage difficile à obtenir aussi proprement sur commande ; elle a sa place dans le rapport.

---

## J5 — Le niveau 2 de l'extension « décision » coûtait plus cher que la campagne

**15 novembre 2026, avant tout entraînement.**

**Symptôme.** `smoke_test.py --time` mesure le débit d'inférence sur T4 et en déduit le coût du niveau 2 aux réglages par
défaut (50 parties de 100 pas, J = 64 suites) : 1,0 h par run de régression, **11,7 h par run de diffusion**, 66,3 h pour les
jetons. Soit 38 h pour les six runs du cœur — davantage que les 28 h de la campagne d'entraînement, et quatre fois la réserve
prévue.

**Cause.** Le coût du niveau 2 n'avait jamais été mesuré, seulement estimé. La planification appelle le modèle J × H × m fois
par pas de jeu ; pour un échantillonneur à 3 pas d'Euler, la facture explose.

**Solution retenue.** J = 32 suites et 20 parties au lieu de 64 et 50, identiques pour tous les runs : coût divisé par cinq,
≈ 7,6 h. Contrepartie assumée : l'erreur type sur les contacts par partie passe de 0,06 à 0,09, ce qui rend H5a **plus
difficile** à confirmer — dans le sens de la sévérité, donc acceptable.

**Statut : décidé, non exécuté.** Tracé comme révision de C7 dans [`DECISIONS.md`](DECISIONS.md). Reste à faire avant le niveau
2 : relancer H0f à ces réglages (`validate_decision.py --games 20 --candidates 32`, sans GPU) pour vérifier qu'un planificateur
affaibli passe encore le contrôle. S'il échoue, garder J = 64 et ne réduire que le nombre de parties.

---

## J4 — Montage de Google Drive depuis l'extension Colab de VS Code

**Symptôme.** `drive.mount('/content/drive')` demande une autorisation servie par la page du notebook ; selon les sessions,
l'appel échoue depuis VS Code avec `MessageError: [dfs_ephemeral] ... unauthorized`.

**Solution retenue.** La cellule 2 tente le montage, et en cas d'échec bascule sur le disque de la session en l'annonçant
explicitement, via les variables `BASE` et `PERSISTANT`. Quand `PERSISTANT` vaut `False`, les résultats disparaissent avec la
session : il faut alors lancer la campagne depuis le navigateur.

**Statut : contourné.** Le montage a finalement fonctionné depuis VS Code ; l'affirmation initiale selon laquelle il était
impossible était trop forte et a été corrigée dans le notebook et dans `notebooks/README.md`. Le repli reste utile : il rend la
cellule robuste au lieu de la faire échouer.

---

## J3 — Avertissement PyTorch sur la perte journalisée

**Symptôme.** Au test de fumée : `UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected
behavior`, émis à chaque journal d'entraînement.

**Cause.** `float(loss)` appelé sur un tenseur encore rattaché au graphe d'autograd.

**Solution retenue et appliquée.** `float(loss.detach())` dans [`trainer.py`](../src/flou/train/trainer.py), ligne 183.

**Statut : résolu.** Sans effet sur les résultats — c'était bien un avertissement, pas une fuite mémoire — mais un journal
propre évite de s'interroger à chaque run.

---

## J2 — Un test de pseudo-terminal échouait sous Jupyter et Colab

**Symptôme.** `test_script_in_a_real_terminal_reads_single_keypresses` échoue avec « Terminal trop petit (80x24) », alors que le
pseudo-terminal ouvert par le test fait bien 100 × 50.

**Cause.** `shutil.get_terminal_size()` donne la **priorité aux variables d'environnement `COLUMNS` et `LINES`** sur la taille
réelle du terminal. Jupyter et Colab les exportent à 80 × 24 ; le jeu refusait donc de démarrer dans un terminal pourtant assez
grand.

**Solution retenue et appliquée.** Le test retire `COLUMNS` et `LINES` de l'environnement du sous-processus
([`tests/test_terminal.py`](../tests/test_terminal.py)).

**Statut : résolu**, vérifié sur Colab : 265 tests, aucun échec.

---

## J1 — Six tests échouaient sous Windows avec `UnicodeEncodeError`

**Symptôme.** Sous Windows 11, six tests échouent avec `UnicodeEncodeError: 'charmap' codec can't encode character`.

**Cause.** Quand sa sortie est redirigée, Python écrit en **cp1252** et non en UTF-8 ; or les scripts affichent des accents, des
lettres grecques (δ, ρ) et des caractères de cadre.

**Solution testée.** Préfixer les commandes de `PYTHONUTF8=1` : 263 tests passent, 2 ignorés, code de sortie 0. Documenté dans
le README (préambule du §3 et tableau de dépannage du §3.8).

**Statut : contourné.** Un correctif de fond a été proposé mais **non appliqué** : deux lignes dans `scripts/_bootstrap.py`
(`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`, idem pour `stderr`). Il rendrait le dépôt utilisable sous Windows
sans précaution, au prix d'une modification du comportement d'affichage de tous les scripts. La décision est en attente,
consignée dans le bloc D du [`TODO.md`](../TODO.md).
