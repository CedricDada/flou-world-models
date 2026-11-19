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

## J11 — Troisième session Kaggle : 10 runs sur 12, une fausse alerte de mon fait, et le balayage de σ

**18 novembre 2026.** Session `STAGE = 'core_control'`, 10,3 h, premier lancement des pièces de J9 et J10 sur GPU. Tout a
tourné sans erreur : garde-fou passé (5 runs évalués dans l'état repris, 5 attendus), bloc 6bis en **16 minutes mesurées**
(réévaluation des 5 anciens runs, référence exacte, balayage de σ), puis l'entraînement jusqu'au budget de 10 h. Sauvegarde OK
(`etat.zip`, 493 Mo). **10 runs sur 12 sont évalués.** `control-regression-p0.00-s0` s'est arrêté proprement au budget et
reprendra ; `control-diffusion-p0.00-s0` reste à faire (environ 5 h pour les deux).

Durées d'entraînement : jetons s1 1,04 h (dont 0,87 h en session 2), régression s2 2,35 h, diffusion s2 2,37 h, jetons s2
0,84 h, oracle 2,33 h — un peu plus rapides qu'en session 2 (2,6 h pour la régression et la diffusion).

### Résultats nouveaux (p = 0,3, ρmax = 0,1599, 2 000 états de test)

| Run | floue | autre | nettes | p̂ [IC 95 %] | au rail : floue / autre / nettes |
|---|---|---|---|---|---|
| régression s2 | 97,9 % | 1,5 % | 0,7 % | — | 99,2 / 0,1 / 0,7 % ; H0b réussi (1,013, fuite z = 0,5) |
| diffusion s2, n = 1 | 0 % | 100 % | 0 % | — | 46,0 / 3,7 / 50,3 % |
| diffusion s2, n = 3 | 3,7 % | 4,7 % | 91,6 % | 0,358 [0,336 ; 0,380] | 3,8 / 1,2 / 95,0 % |
| diffusion s2, n = 20 | 0 % | 3,5 % | 96,5 % | 0,332 [0,311 ; 0,353] | 0 / 0,7 / 99,4 % |
| jetons s1, τ = 1 | 0 % | 68,7 % | 31,3 % | 0,260 [0,226 ; 0,297] | 0 / 19,1 / 80,9 % |
| jetons s2, τ = 1 | 0 % | 39,1 % | 61,0 % | 0,313 [0,287 ; 0,340] | 0 / 12,1 / 87,9 % |
| oracle | 0 % | 1,2 % | 98,8 % | 0,287 [0,267 ; 0,307] | 0 / 0 / 100 % |

**La réévaluation est déterministe.** Les cinq runs réévalués pour porter la clé `rail` redonnent les chiffres de J9 à la
décimale près (p̂ 0,327 et 0,255 à n = 3, rapports MSE inchangés).

**Aperçu, pas un verdict** (c'est `analyze.py` qui tranche) : H2a sera réfutée, comme annoncé en J9 — 0 % de flou à n = 1 sur
les trois graines. H2b ne sera vraisemblablement pas confirmée : à n = 3, la graine s2 donne p̂ = 0,358, à 0,058 de p, au-delà
des 0,05 exigés pour chaque graine (s0 : 0,027 ; s1 : 0,045).

**La mesure au rail sépare bien les deux défauts.** Les « autre » des jetons tombent de 52 / 69 / 39 % à 16 / 19 / 12 % au
rail : l'essentiel venait de l'agent décalé, pas de la bifurcation. La diffusion à n = 1, 100 % « autre » sur l'image
entière, est au rail à moitié floue, à moitié nette (46 à 53 % de flou), sur les trois graines. Les régressions ne bougent
presque pas. Réserve de J10 toujours valable : au rail, ρmax n'est pas recalibré, ces « autre » sont un minorant.

### 1. L'alerte « oracle sous le plancher de Bayes » est fausse, et l'erreur est dans le script

`training_curves.py` a levé : « control-oracle-p0.30-s0 : perte d'entraînement finale 0,000146 SOUS le plancher de Bayes
0,000662 (0,22 x) : le modèle a mémorisé la branche ». Le plancher p(1−p)δ² ne borne qu'un modèle qui **ignore** la branche.
L'oracle la reçoit en entrée (`RegressionModel(oracle=True)`) : passer dessous est exactement son rôle. Tout le reste est sain
— validation plate (+0,8 %, pente −0,008, t = −0,9), 98,8 % d'images nettes, p̂ = 0,287.

**Cause.** La vérification portait sur `("regression", "oracle")`, au motif que les deux familles ont une perte MSE. C'est la
connaissance de la branche qui compte, pas la forme de la perte : l'erreur a été introduite à l'écriture du script (J7), et n'a
pu se voir qu'au premier run d'oracle.

**Solution appliquée.** Le plancher n'est vérifié que pour la régression (`FAMILLES_PLANCHER`), la logique est isolée dans
`alerte_plancher()`, et un test reproduit le cas mesuré : l'oracle à 0,22 fois le plancher ne lève rien, la régression dans la
même situation lève toujours. La règle de pente, elle, n'a rien signalé à tort sur les 11 runs : la diffusion s2 finit 36 %
au-dessus de son minimum, mais sa tendance descend (pente −0,68, t = −5,6) — affiché comme information, comme prévu par J9.

### 2. Le balayage de σ : le réseau suit le débruiteur exact là où σ est couvert, et seulement là

Lecture fixée avant la mesure (DECISIONS.md, complément du 17 novembre). Un pas d'Euler depuis chaque σ, 2 000 états :

| σ | floue s0 / s1 | référence exacte | « autre » s0 / s1 | ρ médian | α̂ p5 – p95 (s0) | référence α̂ | exemples vus (σ ≥) |
|---|---|---|---|---|---|---|---|
| 1 | 0 / 0 % | 7,6 % | 100 / 100 % | 1,29 / 1,34 | 0,11 – 0,68 | 0,00 – 1,00 | — |
| 2 | 0 / 0 % | 14,1 % | 100 / 100 % | 0,48 / 0,53 | 0,00 – 0,97 | 0,00 – 1,00 | — |
| 5 | 34,8 / 46,2 % | 34,0 % | 8,2 / 7,3 % | 0,09 | 0,01 – 0,97 | 0,00 – 0,98 | 21 400 |
| 10 | 53,3 / 56,8 % | 56,1 % | 7,0 / 6,5 % | 0,09 | 0,05 – 0,80 | 0,03 – 0,85 | 3 913 |
| 20 | 0 / 0 % | 75,8 % | 100 / 100 % | 0,21 | 0,11 – 0,65 | 0,11 – 0,60 | 525 |
| 40 | 0 / 0 % | 91,9 % | 100 / 100 % | 0,33 / 0,38 | 0,05 – 0,78 | 0,19 – 0,44 | 51 |
| 80 | 0 / 0 % | 99,7 % | 100 / 100 % | 0,39 / 0,41 | 0,03 – 0,86 | 0,24 – 0,37 | 4 |

(Lignes 30 et 60 omises : elles s'intercalent sans surprise. Référence calculée sur 300 états dans la session.)

**Lecture, exploratoire.** Trois régimes, chacun annoncé à l'avance :
- **σ = 5 à 10, densément couverts** : le réseau reproduit le débruiteur exact — même part de flou à quelques points près,
  même dispersion de α̂, ρ médian 0,09 (sous ρmax). L'échantillonneur à un pas *fonctionne* là.
- **σ ≥ 20, où la couverture s'effondre** : 100 % « autre », ρ médian qui croît avec σ (0,21 → 0,41), et une dispersion de α̂
  qui **s'élargit** quand celle de la référence se resserre. C'est l'hypothèse de J9 : à σmax = 80, vu 4 fois, la sortie n'est
  pas la moyenne conditionnelle.
- **σ ≤ 2** : 100 % « autre » aussi, mais pour la raison anticipée en J10 — le point de départ x = σ·n est loin de la loi
  d'entraînement (85 écarts-types à σ = 1). Ces lignes ne disent rien de la couverture.

Le partage moitié floue, moitié nette de n = 1 au rail (section précédente) est aussi ce que le débruiteur exact donne à
σ ≈ 10 (56 % de flou). C'était l'observation de l'agent en J10, sur 25 images ; elle tient sur 2 000 états et trois graines.
Rien de tout cela ne change H2a : le verdict reste celui de la règle préenregistrée, et ce balayage en donne la raison.

**Statut : résolu pour l'alerte (corrigée, testée), en cours pour la campagne.** La session 4 finit les deux contrôles p = 0
et lance l'analyse (`MIN_EVALUATED = 10`, `RUN_ANALYSIS = True`) ; le balayage, idempotent, ne traitera plus que
`core-diffusion-p0.30-s2`. Pièces : le journal de la session et le notebook exécuté restent sur Kaggle, ainsi que l'état
dans le Dataset `flou-wm-etat` ; ils ne sont pas versionnés ici.

---

## J10 — Relecture du commit 61dfcc6 avant la session 3 : un garde-fou troué, une prédiction fausse

**17 novembre 2026.** Le même agent externe a relu le commit `61dfcc6` (règle de pente, mesure au rail, balayage de σ) avant
qu'on engage dix heures de GPU. Sa note est conservée telle quelle dans
[`incidents/J10_relecture_avant_session3/`](incidents/J10_relecture_avant_session3/). Trois points, tous vérifiés
indépendamment avant d'être appliqués.

**1. Le garde-fou « Dataset périmé » ne couvrait pas le cas qui allait se présenter.** Il ne levait l'erreur que si
`core-regression-p0.30-s0` n'était pas évalué. Or le notebook exporté portait dans ses métadonnées une **version précise** du
Dataset (`sourceId` 22284860), celle de la session 2, dont l'état ne contenait qu'un run évalué : une session 3 lancée sans
« mise à jour de l'entrée » aurait passé le garde-fou et refait les runs 2 à 5, soit environ 10 h de GPU perdues.
**Solution :** compter les `eval.json` contre `MIN_EVALUATED` (5 aujourd'hui), à relever après chaque session — la dernière
ligne de la session affiche la valeur à saisir. Le notebook versionné n'épingle plus aucune version de Dataset.
Testé sur un état à 1 run (erreur levée) et sur l'état réel de la session 2 (5 runs, passe).

**2. La prédiction du balayage de σ était fausse.** La docstring annonçait qu'un pas depuis un σ « bien couvert » (1 à 5)
redonnerait du flou. Même un débruiteur exact ne le ferait pas : le pas part de bruit pur, et dès que σ < ‖A − B‖ ≈ 15 le bruit
seul tranche la branche — image **nette**, ρ = 0. De plus, le point de départ n'est dans la loi d'entraînement qu'à σ ≳ 40 :
il n'existe pas de σ à la fois bien couvert et dont le départ soit crédible, si bien qu'un mauvais résultat à σ ≤ 10 ne
distingue pas « σ mal couvert » de « point de départ hors distribution ». **Solution :** la lecture de référence est fixée
dans `DECISIONS.md` **avant** le balayage (table d'un débruiteur exact, calculée par `scripts/ideal_sigma_reference.py`, NumPy
seul, reproduite sur 2 000 états), la plage passe à 1, 2, 5, 10, 20, 30, 40, 60, 80, et le notebook affiche la référence juste
avant le balayage.

**3. Le seuil ρmax n'est pas recalibré au rail.** Au rail, le même bruit synthétique donne un quantile à 99,5 % de ρ de 0,036 au
lieu de 0,164 (mesuré : 0,0359 contre 0,1645 sur 300 états) : la mesure au rail est 4,6 fois plus tolérante, ses « autre » sont un
minorant. Aucun recalibrage — la mesure reste exploratoire et comparable à J9 — mais la réserve est consignée dans `DECISIONS.md`
et `__rho_rail` est conservé par état pour re-seuiller sans réévaluer.

**Deux améliorations du même passage.** La réévaluation devient idempotente (seuls les runs sans clé `rail` sont refaits) et le
balayage aussi (un run déjà balayé est sauté, `--force` pour refaire) : les deux interrupteurs du notebook restent allumés
sans coût. Et les deux blocs sont désormais protégés par un `try`, parce qu'une exception dans un bloc exploratoire ne doit
pas coûter les dix heures d'entraînement qui suivent.

**Chiffres corrigés au passage.** Le calcul « α̂ de 0,26 à 0,34 » de J9 était un calcul de tête ; recalculé : 0,24 à 0,37. Et
la durée « 17 minutes » annoncée pour les mesures du 17 novembre mélangeait du mesuré (14 minutes de réévaluation) et de
l'estimé (le balayage, non chronométré) ; le notebook le dit désormais.

**Statut : appliqué, non exécuté sur GPU.** Ce qui demande PyTorch (`sigma_sweep.py`, `evaluate.py`, l'entraînement) n'a pas
tourné ici, ni côté agent : seule la logique du notebook a été testée hors Kaggle (syntaxe des 15 cellules de code, garde-fou,
détection des runs à réévaluer, idempotence). La session 3 est la première exécution réelle de ces pièces.

---

## J9 — Deuxième session Kaggle : une fausse alerte, et deux limites de mesure mises au jour

**17 novembre 2026.** Session `STAGE = 'core_control'`, 10 h de GPU : 4 runs évalués (diffusion s0, jetons s0, régression
s1, diffusion s1), plus le run de jetons s1 interrompu à 0,87 h, qui sera repris. Sauvegarde OK (`etat.zip`, 273 Mo).
**5 runs sur 12 sont évalués.** L'analyse ci-dessous a été produite par un agent externe à partir des journaux téléchargés,
puis **recontrôlée indépendamment** : les tableaux, les pentes et les parts d'énergie se reproduisent (voir « Ce qui a été
revérifié » en fin d'entrée).

### Résultats (p = 0,3, ρmax = 0,1599, 2 000 états de test)

| Run | floue | autre | nettes | p̂ [IC 95 %] | MSE / plancher |
|---|---|---|---|---|---|
| régression s0 | 98,2 % | 1,4 % | 0,4 % | — | 1,011 (H0b réussi) |
| régression s1 | 96,7 % | 1,4 % | 1,9 % | — | 1,016 (H0b réussi) |
| diffusion s0, n = 1 | 0 % | **100 %** | 0 % | — | 2,09 |
| diffusion s0, n = 3 | 4,4 % | 4,3 % | 91,2 % | 0,327 [0,306 ; 0,349] | 2,04 |
| diffusion s0, n = 20 | 0 % | 3,8 % | 96,2 % | 0,324 [0,303 ; 0,345] | 2,10 |
| diffusion s1, n = 1 | 0 % | **100 %** | 0 % | — | 2,12 |
| diffusion s1, n = 3 | 5,4 % | 5,5 % | 89,1 % | 0,255 [0,235 ; 0,276] | 1,90 |
| diffusion s1, n = 20 | 0 % | 3,4 % | 96,7 % | 0,250 [0,231 ; 0,270] | 1,96 |
| jetons s0, τ = 1 | 0 % | **51,8 %** | 48,2 % | 0,258 [0,231 ; 0,287] | 3,40 |

Le rapport MSE / plancher vaut environ 2 pour la diffusion, n ≥ 3. C'est la valeur attendue d'un échantillonneur qui tire une
branche indépendamment de la vraie issue : 2p(1−p)δ² (Blau et Michaeli, 2018). La régression est à 1,0. Le cas jouet
prévoyait p̂ ≈ 0,19 à n = 3 (biais de discrétisation) ; le biais observé est bien plus faible, avec une variation d'une graine
à l'autre (0,327 et 0,255).

**Piège de lecture.** Le champ `chk_h0b` de chaque `eval.json` vaut « échoué » pour la diffusion et les jetons. C'est normal
et sans conséquence : H0b demande un rapport dans [1,00 ; 1,10], ce qui n'a de sens que pour un prédicteur de la moyenne, et
`analyze.py` ne calcule le verdict H0b **que sur les runs de régression du cœur** (`analysis.py`, sélection `family="regression"`).
En revanche, la fonction `report_run` du notebook Kaggle affiche « H0b ÉCHOUÉ » pour ces runs, ce qui est trompeur : à corriger
pour n'évaluer H0b que sur la régression et l'oracle.

### 1. L'alerte de surapprentissage sur `core-diffusion-p0.30-s0` est une fausse alerte

`training_curves.py` a signalé « val_loss remontée de +30,2 % depuis son minimum (pas 34 000) ». La règle compare le
**dernier** point de validation au minimum. Pour la régression, la validation est lisse, et la règle a bien détecté J6.
Pour la diffusion, la validation fluctue fortement d'un point à l'autre, même à bruit fixe (`fixed_noise`), car les poids
eux-mêmes varient :
- s0 : +101 % au pas 14 000 (0,00140 → 0,00282), puis la baisse reprend ;
- s1 : +80 % au pas 32 000, puis 0,000375 au pas 34 000.

Le +30 % final de s0 est un point de ce bruit. Une régression linéaire de log(val) sur la seconde moitié de l'entraînement
tranche nettement. Les deux colonnes de droite sont un recalcul indépendant, avec un découpage de la seconde moitié
légèrement différent : les valeurs bougent, la conclusion non.

| Run | pente (agent) | t (agent) | pente (contrôle) | t (contrôle) |
|---|---|---|---|---|
| J6 (100 000 transitions, surapprentissage avéré) | **+0,228** | **+12,9** | **+0,214** | **+10,7** |
| diffusion s0 | −0,425 | −5,2 | −0,399 | −3,9 |
| diffusion s1 | −0,361 | −2,2 | −0,341 | −1,6 |
| régression s0 | −0,002 | −0,9 | −0,001 | −0,3 |
| régression s1 | −0,001 | −0,2 | −0,003 | −0,5 |
| jetons s0 (gpt / vq) | −0,180 / −1,049 | −9,0 / −2,5 | identique | identique |
| jetons s1 (gpt / vq) | −0,301 / −2,050 | −16,0 / −4,6 | identique | identique |

Pentes par 10 000 pas. La validation de la diffusion s0 **baisse encore** en fin d'entraînement : c'est l'inverse d'un
surapprentissage. Les mesures d'évaluation (p̂ calibré, MSE ≈ 2 fois le plancher) ne montrent rien d'anormal. Décision : la
campagne continue.

**Correctif proposé pour `scripts/training_curves.py`.** Lever l'alerte quand la pente de log(val) sur la seconde moitié est
positive avec t > 3, et garder l'ancienne règle comme simple information. Sur les runs ci-dessus, la nouvelle règle signale J6
et seulement J6. *Réserve :* le seuil t > 3 est choisi après avoir vu ces sept runs, sur 9 points de validation chacun (4 à 6
pour les stades des jetons). C'est un garde-fou de surveillance, pas un verdict préenregistré — il ne change aucune hypothèse.

### 2. La diffusion à n = 1 n'est pas « la moyenne » : 100 % « autre », sur les deux graines

On attendait, d'après le cours, un échantillon en un pas égal à D(x ; σmax), c'est-à-dire la moyenne conditionnelle : une image
floue, α̂ ≈ p, ρ ≈ 0, MSE ≈ 1 fois le plancher. C'est d'ailleurs ce qu'affirme la docstring de `karras_sigmas`. On observe
autre chose : 100 % « autre », MSE ≈ 2,1 fois le plancher.
- `samples_n1.png` montre l'ennemi dessiné **à moitié** aux deux positions, ce qui est bien un mélange. Mais toute l'image est
  couverte d'un **grain** de fond, et le poids du mélange **change d'un tirage à l'autre** : α̂ va de 0,03 à 0,86 (5e et 95e
  centiles), centré sur 0,29.
- Sur les images conservées, 87 % de l'énergie de l'écart ‖ŷ − (α̂A + (1−α̂)B)‖² tombe hors du rail de l'ennemi et hors de
  l'agent : c'est le grain. Restreint au rail de l'ennemi, le classement donne, sur 25 images, 14 « floue », 9 « nette-B »
  et 2 « nette-A » pour s0 (9, 14 et 2 pour s1).

**Hypothèse, à confirmer.** À l'entraînement, σ suit une loi log-normale (P_mean = −1,2, P_std = 1,2). Sur 2 227 200
exemples, **4** avaient σ ≥ 80 et **525** avaient σ ≥ 20 (recalculé : P(σ ≥ 80) = 1,65·10⁻⁶). Le débruiteur n'a donc
pratiquement pas appris à σmax = 80, et sa sortie n'y est pas la moyenne conditionnelle. Le mécanisme est cohérent avec le
préconditionnement d'EDM : à σ = 80, c_in ≈ 1/80 et c_skip ≈ 4·10⁻⁵, donc l'entrée du réseau est du bruit presque pur et sa
réponse dépend du tirage. Le débruiteur exact ferait varier α̂ très peu d'un tirage à l'autre : de 0,24 à 0,37 (5e–95e centiles,
recalculé sur les 2 000 états, ‖A−B‖ médiane 15,3 à l'échelle du modèle ; la première version de cette entrée
donnait 0,26 à 0,34, un calcul de tête sur ‖A−B‖ ≈ 14). À n ≥ 3, les pas suivants corrigent ce premier pas. **Mais
cette lecture est incomplète aux σ petits : voir J10.**

**Conséquence.** H2a (n = 1 plus floue que n = 3 d'au moins 10 points) sera réfutée, pour une raison étrangère à la question
du projet : elle mesure la couverture de σ à l'entraînement. On garde H2a telle que préenregistrée et on rapporte sa
réfutation avec cette explication. Diagnostic exploratoire proposé, à déclarer comme tel dans `DECISIONS.md` : échantillonner
en un pas depuis σ ∈ {1, 2, 5, 10, 20, 80}, sur les débruiteurs existants, sans réentraînement.

### 3. Les jetons : la moitié des images « autre » vient de l'agent, pas de la bifurcation

51,8 % des images du run de jetons s0 sont classées « autre ». Les images (`samples_tau1.png`) sont nettes, mais l'**agent**
y est souvent décalé d'un ou deux pixels, ou dédoublé au bord d'un jeton. Sur les 25 images « autre » conservées, 58 % de
l'énergie de l'écart tombe sur l'agent. Restreint au rail de l'ennemi, le classement donne 19 images nettes (4 A, 15 B) et
6 « autre ».

C'est la limite connue du classifieur sur l'image entière, déjà mesurée sur le mélange idéal : un agent décalé d'un pixel
donne ρ ≈ 0,39. ρ est divisé par δ, qui ne mesure que le déplacement de l'ennemi. Pour la régression, les rares « autre »
viennent de la même façon du reste de l'image (95 % de l'énergie), et restreints au rail ils sont « floue ».

**Mesure proposée (exploratoire, à déclarer avant l'analyse).** En plus du classifieur sur l'image entière, qui reste la
mesure principale, préenregistrée, calculer α̂ et ρ restreints au rail de l'ennemi. Pixels où A et B diffèrent, dilatés de
2 pixels. Cette mesure sépare deux questions : « la bifurcation est-elle tranchée, mélangée ou ratée ? » et « le reste de
l'image est-il juste ? ». Un prototype NumPy existe ([`scripts/where_other.py`](../scripts/where_other.py)) ; il ne demande
ni GPU ni réentraînement.

*Base de mesure à garder en tête :* `eval_arrays.npz` ne conserve que **25 images par classe**. Les parts d'énergie et les
reclassements ci-dessus portent donc sur 25 cas, pas sur les 2 000 états. Ils suffisent à établir l'origine de l'écart, pas à
chiffrer une proportion.

### Ce qui a été revérifié

Sur les journaux et les `eval.json` téléchargés, avec un code écrit indépendamment de celui de l'agent :
- le tableau de résultats, ligne à ligne, y compris les variantes n = 5 que l'entrée ne reprend pas ;
- les pentes de log(val) et leurs t (colonnes « contrôle » ci-dessus) : même signe, même conclusion pour les sept runs ;
- le dénombrement des σ vus à l'entraînement : 4 et 525, aux valeurs annoncées ;
- les parts d'énergie et les reclassements au rail, par exécution de `where_other.py` : 87,6 % hors rail pour la diffusion
  n = 1 (14 floue / 9 nette-B / 2 nette-A pour s0, 9 / 14 / 2 pour s1), 57,6 % sur l'agent pour les jetons (19 nettes, 6
  autre), 94,8 % sur le reste pour la régression (25 « autre » reclassées « floue »).

Deux défauts du prototype ont été corrigés au passage : les cibles par défaut employaient les clés `n1`, `n3`, `tau1` alors
que les variantes s'appellent `n=1`, `n=3`, `tau=1` — le script annonçait « pas d'images conservées » sans rien signaler — et
ρmax était codé en dur au lieu d'être lu dans `reports/instrument.json`.

**Statut : résolu pour les trois propositions, en cours de vérification pour la campagne.** Les trois ont été retenues et
consignées le 17 novembre dans [`DECISIONS.md`](DECISIONS.md), avant toute analyse, puis appliquées :

| Proposition | Où elle vit | Vérification |
|---|---|---|
| Règle de pente | `scripts/training_curves.py`, option `--seuil-t` | signale J6 (pente +0,21, t = +10,7) et aucun des six runs sains ; trois tests |
| Balayage de σ | `scripts/sigma_sweep.py` (nouveau) | écrit, **pas encore exécuté** : demande un GPU et les poids |
| Classifieur au rail | `classifier.rail_mask` / `alpha_rho_rail`, clé `rail` de chaque `eval.json` | quatre tests, dont une image salie hors du rail qui passe de « autre » à « nette-A » |

Reste à faire, dans l'ordre : réévaluer les cinq runs déjà évalués (`evaluate.py --all --force`, aucun réentraînement) pour
qu'ils portent tous la mesure au rail, lancer le balayage de σ sur les deux runs de diffusion, puis poursuivre la campagne —
reprise du run de jetons s1, puis régression s2, diffusion s2, jetons s2 et oracle (environ 9,5 h), les deux contrôles p = 0
dans une dernière session. Le notebook Kaggle porte les deux interrupteurs `RUN_REEVAL` et `RUN_SIGMA_SWEEP`.

---

## J8 — Colab libérait la T4 à chaque mise en veille : la campagne passe sur Kaggle

**16 novembre 2026.**

**Symptôme.** Avec Colab Pro (T4 garantie, sans l'option Pro+), l'entraînement s'arrêtait dès que l'ordinateur se mettait en
veille ou que la fenêtre se fermait : le runtime était libéré et la T4 rendue. Le run `core-regression-p0.30-s0` (400 000
transitions, après J6) n'avançait donc que par morceaux. Les checkpoints sur Drive ont évité toute perte : on lit
« reprise : stade 0, pas 10000, 0.77 h déjà consommées » dans une session, puis le checkpoint du pas 28 000 (2,18 h) dans la
suivante.

**Cause.** D'après les sources consultées (non vérifié par un test), l'exécution en arrière-plan est réservée à Colab Pro+ et
au-delà ; avec Pro, le runtime reste lié à l'interface ouverte. Les checkpoints rendent un run reprenable, pas autonome : il
fallait garder la machine éveillée pendant des dizaines d'heures.

**Options examinées.**

| Option | Pour | Contre |
|---|---|---|
| Colab Pro+ (49,99 $/mois) | aucun changement de code | abonnement, coût |
| Location de GPU (RunPod, Vast.ai, Lambda…) | machine dédiée, `tmux` ou `nohup` | payant, administration, oubli d'extinction facturé |
| **Kaggle** | gratuit, T4 x2, « Save & Run All (Commit) » tourne en arrière-plan | quota hebdomadaire, 12 h par session, Drive non montable |

**Le quota, vérifié.** Kaggle annonce environ 30 h de GPU par semaine (sa page d'aide dit « 30 hours or sometimes higher
depending on demand and resources »), et ajoute **15 h par semaine** aux comptes liés à un abonnement Colab Pro — 30 h pour
Pro+ — sous forme de promotion expérimentale, révocable sans préavis. Le compte utilisé étant Pro, la réserve attendue est de
**45 h par semaine**. Le chiffre qui fait foi est celui affiché dans les réglages du compte Kaggle, pas ces annonces.

**Ce que cela change pour le calendrier.** Le plan annonce 38 h pour les 25 runs, estimation produite par `plan.py` à partir
des temps du test de fumée. Le seul run réellement chronométré, `core-regression-p0.30-s0`, a demandé **2,7 h** pour
34 800 pas — et la régression est la famille la moins chère. Les 12 runs prioritaires (4 régressions, 4 diffusions, 3 jetons,
1 oracle) coûteront donc **au moins 32 h**, vraisemblablement davantage : environ une semaine de quota. Les 25 runs du plan
complet dépassent nettement les 38 h annoncées. À recalculer avec `plan.py --timing` dès qu'un run de diffusion et un run de
jetons auront été chronométrés.

**Solution retenue : Kaggle, avec l'état du projet dans un Dataset privé.**

1. Le dossier `flou_wm` de Drive a été compressé en `etat.zip` (`runs/` et `reports/`), puis envoyé comme Dataset Kaggle privé
   `cedricdada/flou-wm-etat` (66 Mo). `data/` n'y est pas : la génération est déterministe, et deux générations successives
   donnent des tableaux identiques (vérifié avec NumPy). Les données sont donc régénérées à chaque session, en une minute.
2. Nouveau notebook [`notebooks/flou_kaggle.ipynb`](../notebooks/flou_kaggle.ipynb). Une session est un commit Kaggle :
   - `restore_state()` copie l'état du Dataset dans `/kaggle/working/flou_wm` ;
   - clonage du dépôt, données, `validate_instrument.py` ;
   - entraînement selon `STAGE` : `first_run` termine et évalue seulement `core-regression-p0.30-s0` ; `core_control`
     lance `run_plan.py --groups core control --max-hours 10` ;
   - contrôle du surapprentissage : `training_curves.py`, H0b, flou de la régression, puis un verdict « feu vert » ou « ne pas
     lancer le cœur » ;
   - `save_state()` publie une nouvelle version du Dataset avec le Kaggle CLI. Le jeton API est rangé dans un secret Kaggle
     (`flou-wm`), jamais dans le code.
3. `/kaggle/working/flou_wm` devient aussi la sortie de chaque version : c'est un filet de sécurité si l'envoi au Dataset
   échoue.

**Écueils rencontrés en route.** Chacun est évité dans le notebook.

| Écueil | Correction |
|---|---|
| Accélérateurs grisés (seul « None » disponible) | Vérification du compte par téléphone |
| `git clone` : « Could not resolve host: github.com » | Internet coupé par défaut : activer *Internet On* |
| `drive.mount` lève `NotImplementedError` hors de Colab | Remplacé par le Dataset |
| Archive au format zip, pas tar | `zipfile` au lieu de `tarfile` |
| Chemin réel du Dataset : `/kaggle/input/datasets/cedricdada/flou-wm-etat/etat` | Recherche récursive de `runs/` sous `/kaggle/input` |
| La cellule du notebook Colab qui archive le run à 100 000 transitions puis vide `runs/` (le `shutil.move` suivi des `rmtree`, juste après la génération des données de la section 4) aurait effacé le run en cours si elle avait été relancée | Absente du notebook Kaggle |
| Un commit reprend la version du Dataset attachée au notebook | Mettre l'entrée à jour avant chaque session |

**Vérifié sur Kaggle**, en sessions interactives :
- tests NumPy passés et test de fumée entièrement `[OK]` sur T4, avec PyTorch 2.11.0+cu130 ;
- ρmax = 0,1599 et H0a réussi, avec les 2 000 états de test (J7) ;
- reprise du run au pas 28 000 confirmée (« reprise : stade 0, pas 28000, 2.18 h déjà consommées ») ;
- authentification du CLI correcte (`kaggle datasets list --mine`).

La cellule de contrôle a aussi été testée hors GPU, sur un régresseur simulé (verdict « feu vert ») et sur un journal
reproduisant J6 (verdict « ne pas lancer », deux alertes).

**Réserve à mentionner dans le rapport.** Le run `core-regression-p0.30-s0` a été entraîné jusqu'au pas 28 000 sur Colab
(PyTorch 2.14.1), puis terminé sur Kaggle (PyTorch 2.11.0), sur le même modèle de GPU (T4). Le changement de version ne remet
pas en cause la validité du run, mais il empêche de le reproduire au bit près. Tous les runs suivants tournent entièrement sur
Kaggle. Une seule des deux T4 est utilisée : le quota se compte en heures de session, pas en heures de carte, donc faire
tourner deux runs en parallèle (`CUDA_VISIBLE_DEVICES=0` et `=1`, par exemple `--groups core` d'un côté et `--groups control`
de l'autre) diviserait par deux le quota consommé par la campagne. Piste non essayée, à peser contre le risque de saturer la
mémoire ou les processeurs de la session.

**Statut : résolu.** Le premier commit (`STAGE = 'first_run'`) a tourné de bout en bout le 16 novembre 2026. Les quatre
points en attente sont levés : reprise au pas 28 000, run terminé (2,67 h cumulées sur T4, 5 872 771 paramètres),
évaluation faite en 16 s, et publication de `etat.zip` (44 Mo) vers le Dataset **depuis le mode batch** — le secret Kaggle
est donc bien lisible hors session interactive. Le verdict de surapprentissage est au vert (J6), ce qui autorise le passage
à `STAGE = 'core_control'`.

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

**Statut : résolu.** Le run relancé sur 400 000 transitions s'est terminé le 16 novembre 2026 et satisfait les deux critères
fixés d'avance. La validation ne remonte plus : minimum 0,000771 au pas 30 000, 0,000775 à l'arrivée, soit **+0,5 %**, dans le
bruit — contre +49 % au run fautif. Et `ratio_expected` vaut **1,011**, au cœur de la bande [1,00 ; 1,10], contre 1,47 ; la
fuite reste nulle (z = 0,7, alerte à 3). H0b est donc **réussi**. La réserve ci-dessus est levée : l'atterrissage dans la bande
a bien eu lieu. Le contrôle a été rendu automatique entre-temps — `scripts/training_curves.py` trace les courbes et lève une
alerte avec un code de sortie non nul (J7) ; il a conclu « aucun symptôme de surapprentissage ».

Le gain sur l'hypothèse centrale est net : le flou de la régression passe de **22,5 % à 98,2 %** des 2 000 images de test, et
« autre » de 2,0 % à 1,4 %. Côté régression, H1 est confirmée très au-delà de son seuil (50 %). Avec 0,4 % d'images nettes,
p̂ n'est plus estimable sur cette famille (8 images, intervalle [0,000 ; 0,369]) : c'est le comportement attendu d'un
prédicteur qui produit la moyenne, et cela ne touche pas H2b, dont le verdict est défini sur la diffusion à n = 3.

Le passage de 100 000 à 400 000 transitions est donc **le correctif qui a sauvé l'hypothèse centrale du projet** : sans lui,
la régression aurait été mesurée comme produisant des images nettes mémorisées, et H1 aurait été réfutée pour une raison qui
n'avait rien à voir avec ce qu'elle prétend mesurer.

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
