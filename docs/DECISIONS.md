# Décisions prises avant le premier entraînement

**Date : 15 novembre 2026.** Aucun entraînement n'a eu lieu à cette date, aucun résultat sur un modèle n'existe : tout ce
document précède les mesures. C'est son unique raison d'être — une réfutation annoncée d'avance est un résultat, une
hypothèse ajustée après coup n'en est pas un.

Ce fichier tranche les décisions laissées ouvertes dans [`../TODO.md`](../TODO.md) (bloc C). Les énoncés et les seuils qu'il
rappelle viennent de [`EXPERIMENTS.md`](EXPERIMENTS.md) §4 et de [`EXTENSION_DECISION.md`](EXTENSION_DECISION.md) §5 ; les choix
d'interprétation y sont marqués **[choix]**. En cas de désaccord avec le dossier initial, le cours préparatoire (chap. 7 et 9)
fait foi.

## Sur quoi ces décisions sont fondées

Trois règles, appliquées uniformément :

1. **La position par défaut est de garder le seuil pré-enregistré**, même quand on a de bonnes raisons de croire qu'il ne sera
   pas atteint. Un choix figé avant les runs ne coûte rien ; un choix déplacé après avoir vu les chiffres retire toute valeur au
   protocole entier, y compris aux hypothèses qu'on n'a pas touchées.
2. **À choix égal, retenir l'option qui rend l'hypothèse plus difficile à confirmer.** C'est déjà le raisonnement du dépôt
   lorsqu'il retient le lecteur doux pour H5a : c'est le lecteur le plus favorable à la régression, donc le test le plus sévère.
3. **On ne change un réglage que s'il mesure autre chose que ce qu'on croit**, jamais parce qu'il est défavorable.

| | Objet | Décision |
|---|---|---|
| C1 | Où est le « modèle du monde » ? | Extension « décision » implémentée (risque estimé, puis planification) |
| C2 | H2b, calibration de la diffusion | Seuil gardé ; la courbe p̂(n) est publiée avec le verdict |
| C3 | Sept choix d'interprétation | Confirmés, avec deux réserves écrites dans le rapport |
| C4 | Puissance statistique | 3 graines acceptées ; k = 3 comparaisons simultanées fixé d'avance |
| C5 | Budget et ordre de sacrifice | 28 h fermes + ~10 h de réserve ; axe p seulement en bonus |
| C6 | H5a, planification sous modèle flou | Gardée telle quelle, lecteur doux, sans second lecteur au niveau 2 |
| C7 | Seuils et réglages de l'extension | Confirmés ; marge de H0f vérifiée sur une seconde graine |
| C8 | Récompense lue dans les pixels | Assumée, avec le périmètre des conclusions explicitement borné |

Les **révisions datées** sont signalées en place quand elles précèdent tout entraînement (C7, réglages du
niveau 2) et en fin de document quand elles suivent un run (volume des données d'entraînement). Le détail des
incidents qui les ont motivées est dans le journal de bord : [`JOURNAL.md`](JOURNAL.md).

---

## C1 — Où est le « modèle du monde » ?

**Ce qui était en jeu.** Les trois familles comparées (régression, diffusion EDM, jetons discrets) ne sont que la *composante
de prédiction* d'un modèle du monde : elles apprennent p(image suivante | images passées, action). Sans usage en aval, le projet
ne pouvait rien dire de la question « le flou est-il un problème ? », seulement « le flou existe-t-il ? ».

**Décision.** L'extension « décision » est implémentée et fait partie du projet : niveau 1, estimation d'un risque (H4) ;
niveau 2, planification dans le modèle (H5). Restent hors du dépôt une récompense apprise et un agent entraîné par
renforcement dans le modèle — DIAMOND annonce ≈ 2,9 jours de RTX 4090 par partie et par graine, hors de portée d'un T4.

**Pourquoi.** Mesurer qu'une prédiction est floue ne coûte rien à personne tant que rien ne s'en sert. Les deux niveaux
retenus transforment la question esthétique en question d'usage sans aucun ré-entraînement : ils réutilisent les modèles du
plan de 25 runs.

---

## C2 — H2b : la diffusion reproduit-elle la fréquence des branches ?

**L'hypothèse.** H2b : la diffusion à n = 3 pas d'Euler reproduit p à ± 0,05, sur au moins 1 000 images nettes et à chaque
graine. **Réfutée si l'erreur atteint 0,05 à chaque graine.**

**Ce qui est en jeu.** Sur un modèle jouet — mélange de deux points, débruiteur **exact**, une dimension, reproduit par
`flou.models.edm_math` — l'échantillonneur d'Euler donne, pour p = 0,3 : p̂ ≈ 0,19 à n = 3 (erreur ≈ 0,11), 0,23 à n = 5,
0,27 à n = 20, 0,28 à n = 50. Autrement dit **un débruiteur parfait échouerait probablement à H2b à n = 3**, et la cause serait
la discrétisation du solveur, pas l'entraînement. Trois options : garder le seuil ; changer d'échantillonneur ou de n de
référence ; relever le seuil.

**Décision.** H2b est gardée telle quelle, avec n = 3 comme point de comparaison. Le verdict sera **lu conjointement avec la
courbe p̂(n)** (`calibration_par_n.csv`, `fig_calibration.png`), qui est produite d'office pour n ∈ {1, 3, 5} plus la référence
n = 20 inscrite au plan.

**Pourquoi.** Relever le seuil maintenant reviendrait à ajuster l'hypothèse à la réponse attendue, c'est-à-dire à faire
exactement ce que la pré-inscription des seuils sert à empêcher (règle 1). Surtout, le point décisif est que **l'échantillonnage
ne coûte aucun entraînement** : le réseau EDM est entraîné une fois et peut être échantillonné à n'importe quel nombre de pas.
La courbe p̂(n) est donc gratuite, et une H2b réfutée devient un résultat quantitatif — « la netteté s'obtient dès n = 3, la
calibration demande davantage de pas » — plus informatif qu'une H2b confirmée.

**À écrire dans le rapport.** Que le calcul du modèle jouet est une **idéalisation** (une dimension, débruiteur exact) : c'était
une raison de s'attendre à une réfutation, pas une prédiction pour le U-Net entraîné. Ne jamais conclure « la diffusion ne
calibre pas » sans la courbe en regard.

---

## C3 — Les sept choix d'interprétation

**Ce qui est en jeu.** Le cours fixe le nombre de contrôles et les hypothèses, pas tous les détails. Sept lectures ont dû être
arrêtées pour que le code existe ; aucune n'est imposée par l'énoncé.

**Décision.** Les sept sont confirmées telles quelles :

1. **Les trois contrôles** = régresseur *oracle* (il reçoit la branche tirée), régression à p = 0, diffusion à p = 0 évaluée à
   n = 1.
2. **La « grille »** = grille de jetons 16 × 16 à K = 256, une seule graine.
3. **ρmax** réglé sur du bruit synthétique σ = 0,01, quantile 99,5 % : ρmax ≈ 0,16.
4. **H0b** évalué sur le rapport MSE/plancher **en espérance**, attendu dans [1,00 ; 1,10].
5. **Qualité à l'horizon 16** = part des rollouts où l'agent *et* l'obstacle sont suivis à ≤ 3 px.
6. **Réanimation des codes morts** du VQ activée ; **pas de moyenne EMA** par défaut.
7. **Budget apparié** en images vues (pas × lot) ; famille C répartie 40 % VQ-VAE / 60 % Transformer.

**Pourquoi.** Aucune de ces lectures ne mesure autre chose que ce qu'elle annonce (règle 3). Deux méritent un mot. Le choix 4
est le seul qui ne pénalise pas un échantillonneur correct : par construction, un échantillonneur parfait a un rapport
MSE/plancher de 2 — la MSE ne distingue donc pas « net et bien calibré » de « flou », d'où l'existence du classifieur. Le
choix 6 évite qu'une famille bénéficie d'un réglage que les autres n'ont pas.

**Les deux réserves à écrire dans le rapport.**

- **ρmax est une convention, pas une mesure.** La part de la classe « autre » dépend de ce réglage et ne doit jamais être lue
  comme un taux absolu. `reports/instrument.json` fournit la sensibilité à × 0,5 / × 1 / × 2 : elle doit figurer dans le
  rapport. Repère utile : un petit objet détruit donne ρ ≈ 0,325 sans bruit, ce qui impose ρmax < 0,32.
- **L'axe K n'est pas un axe pur.** K = 64 pèse 5,91 M paramètres, K = 256 pèse 6,04 M et K = 1 024 pèse 6,59 M (+ 12 % par
  rapport à la régression, 5,87 M). Si H3a est confirmée, la capacité a varié en même temps que la taille du dictionnaire. Le
  run « grille » existe pour séparer l'effet de la finesse de grille de celui de K ; le confondant doit être nommé, pas tu.

---

## C4 — Puissance statistique à trois graines

**Les hypothèses concernées.** Celles dont le verdict repose sur une comparaison entre graines : **H3a** (rappel du petit objet,
K = 64 contre K = 1 024, non appariée, seuil 2,27 écarts-types poolés), **H4a** et **H5a** (régression contre diffusion n = 3,
non appariées, même seuil).

**Ce qui est en jeu.** Avec 3 graines contre 3, la puissance n'est que d'environ **46 %** pour un effet de 2 écarts-types. Les
seuils à très peu de graines sont 2,27 s (non apparié) et 2,48 s_d (apparié) ; avec k comparaisons simultanées ils doivent être
relevés par Bonferroni : 3,23 / 3,96 / 4,57 s pour k = 3 / 6 / 10.

**Décision.** Trois graines sont acceptées. Deux engagements l'accompagnent :

- **k = 3 comparaisons simultanées est fixé maintenant** (H1, H2a, H3a), ce qui donne un seuil de **3,23 écarts-types** pour les
  comparaisons non appariées du cœur. Les hypothèses exploratoires (H3b, H4c, H5b) ne sont pas comptées : elles sont rapportées
  sans seuil.
- **Un résultat sous le seuil est « non concluant », jamais « pas d'effet ».** C'est déjà la règle appliquée par
  `flou.eval.analysis.verdicts` ; elle doit l'être aussi dans la rédaction.

Si du budget reste à la fin du mois, il va en **graines supplémentaires sur l'axe K**, pas sur l'axe p.

**Pourquoi.** La faible puissance n'est pas uniformément gênante. H1 se mesure sur 1 000 états de test, avec des seuils très
écartés (flou > 50 % contre nettes > 90 %) : l'effet attendu est massif. H2a est **appariée** — mêmes débruiteurs, seul le
nombre de pas change — donc bien plus puissante à graines égales. Seules H3a, H4a et H5a sont réellement limitées par le nombre
de graines, et c'est H3a qui porte une hypothèse du cœur : d'où l'affectation du budget résiduel. Choisir k après avoir vu les
résultats serait le biais que Bonferroni est censé corriger ; d'où le fait de l'écrire ici.

---

## C5 — Budget et ordre de sacrifice

**Ce qui est en jeu.** Le plan compte 25 runs. À l'estimation du cours (≈ 1,5 h par run, **jamais mesurée**), les groupes
cumulent : cœur + contrôles (12 runs) ≈ 18 h ; + axe K (6) ≈ 27 h ; + grille (1) ≈ 28,5 h ; + axe p (6) ≈ 37,5 h. S'y ajoute le
niveau 2 de l'extension, de **durée inconnue**.

**Décision.** Le plan engagé est **cœur + contrôles + axe K + grille, soit 19 runs ≈ 28 h**, avec une réserve d'environ 10 h
pour le niveau 2 de l'extension. L'**axe p est sacrifié d'avance** (`--skip-groups p_axis`) et ne sera lancé que s'il reste du
budget après l'analyse et l'extension. L'ordre de priorité du plan (cœur et contrôles, puis axe K, puis grille, puis axe p) est
inchangé.

**Pourquoi.** L'axe p est explicitement exploratoire (une graine, p ∈ {0,1 ; 0,5}) et ne porte aucun verdict : il illustre le
flou en fonction de p. Le sacrifier d'avance, plutôt qu'en catastrophe à la fin, évite de découvrir à mi-parcours qu'il a mangé
le budget du niveau 2, qui lui porte H5. Le chiffre devient ferme après `smoke_test.py --time` et
`plan.py --timing reports/timing.json --budget-hours`, qui donnent le nombre de pas par run réellement finançable.

**Mesuré le 15 novembre 2026** (`smoke_test.py --time`, Tesla T4, PyTorch 2.14) : 0,233 s/pas pour la régression, 0,239 pour la
diffusion, 0,094 + 0,071 pour les deux stades des jetons, en lot 64. Le plan à 20 000 pas ne coûte que 22,4 h, et
`--budget-hours 38` donne **34 800 pas par run** — soit très exactement les **28 h** engagées ici pour 19 runs. C'est donc
cette valeur qui est retenue, et le « ≈ 1,5 h par run » du cours se révèle juste à ce nombre de pas.

**À savoir sur la ressource.** Sur Colab Pro, le plafond réel n'est pas le temps mais les **unités de calcul** : 100 unités par
mois représentent de l'ordre de 57 à 84 h de T4 selon les mesures publiées, taux qui fluctue. Les 28 h engagées plus la réserve
tiennent dans un mois ; 37,5 h plus le niveau 2 tiendraient plus difficilement, sans marge pour reprendre une erreur. Refuser un
L4 ou un A100 s'ils sont proposés : ils consomment trois à six fois plus par heure. Le budget étant apparié en **pas** et non en
temps, changer de GPU en cours de route ne casserait pas la comparaison, mais invaliderait l'estimation de durée.

---

## C6 — H5a : le planificateur à modèle flou perd-il des points ?

**L'hypothèse.** H5a : un planificateur qui utilise la **régression** comme simulateur subit au moins **0,5 contact de plus par
partie** qu'un planificateur utilisant la diffusion à n = 3, écart devant dépasser 2,27 écarts-types poolés (3 graines contre 3).
**Réfutée si l'écart est inférieur à 0,5.** Lecteur de récompense : le lecteur **doux** (affine en l'image).

**Ce qui est en jeu.** Mesuré sans aucun modèle entraîné, sur le vrai jeu (p = 0,3) : une image moyenne *exacte* lue par le
lecteur doux donne le bon risque (erreur 0,001) et le **meilleur** évitement, 0,02 contact par partie — contre 0,16 pour le vrai
jeu échantillonné. Lue par le lecteur à seuil, la même image moyenne se trompe de 0,30 et subit 0,60 contact. Donc si la
régression se comportait comme une moyenne exacte, **H5a serait réfutée**.

**Décision.** H5a est gardée telle quelle, avec le **lecteur doux** et le seuil de 0,5 contact. Le niveau 2 **n'est pas doublé**
avec le lecteur à seuil.

**Pourquoi.** Le lecteur doux est le plus favorable à la régression, donc le test le plus sévère pour H5a (règle 2) : une
confirmation dans ces conditions serait un résultat fort, et une réfutation serait le résultat de **linéarité** — quand le coût
est affine en l'image, le flou ne nuit pas à la décision. Les deux issues sont publiables, ce qui est exactement la situation
recherchée. H5a ne peut être confirmée que si la régression réelle se dégrade en se nourrissant de ses propres images floues sur
huit pas imaginés ; seule l'expérience le dira. Enfin, le contraste entre les deux lecteurs est **déjà** mesuré gratuitement au
niveau 1 par le couple H4a / H4b : le payer une seconde fois au niveau 2 coûterait 2,6 millions d'images par run pour la
régression et 10,2 millions pour la diffusion, sans apporter d'information nouvelle.

---

## C7 — Seuils et réglages de l'extension « décision »

**Les hypothèses concernées.** **H0e** (contrôle) : le lecteur de contact retrouve l'événement du jeu sur ≥ 99 % des vraies
images. **H0f** (contrôle) : avec le vrai jeu comme modèle, le planificateur subit ≤ 0,5 contact par partie et marque ≥ 3 points
de plus que le bot aléatoire. **H4a** : écart ≥ 0,10 entre régression et diffusion, lecteur à seuil, k = 1. **H4b** : la
régression fait au moins aussi bien à 0,02 près, lecteur doux ; réfutée au-delà de 0,05.

**Ce qui est en jeu.** H0e et H0f sont des **verrous** : s'ils échouent, aucun modèle n'est mesuré. Les seuils de H4 viennent de
PETS (arXiv:1805.12114, §5.2 et §7.2) et le protocole du planificateur de PlaNet (arXiv:1811.04551, §2 et Table 1), mais les
valeurs numériques sont des **[choix]**. H0f passe de peu : +3,16 points pour un minimum de 3.

**Décision.** Tous les seuils et réglages sont confirmés : 0,10 (H4a) ; 0,02 et 0,05 (H4b) ; 0,5 contact (H5a) ; H0f inchangé ;
planificateur J = 64 suites, H = 8 pas, m = 4 tirages, λ = 0,5, 50 parties, lecteur doux, p = 0,3 seulement ; 3 000 états au
niveau 1 (≈ 260 états à risque ambigu à k = 1), jeu de test distinct du jeu de test principal.

**Pourquoi.** Le critère décisif de H0f, le nombre de contacts, passe largement : 0,16 pour un maximum de 0,5. C'est le second
critère, le gain sur le bot aléatoire, qui est serré, et il l'est par construction : le score du bot aléatoire varie de −2 à
−3,3 selon les parties. Deux éléments lèvent le doute sur sa fragilité. Le contrôle a été exécuté deux fois, sur deux systèmes
d'exploitation différents (Linux puis Windows 11), avec des résultats **identiques à la deuxième décimale** : 100 % d'accord pour
H0e sur 30 000 images dont 887 contacts, planificateur à 1,10 point et 0,16 contact par partie, bot aléatoire à −2,06, bot expert
à 1,66, gain +3,16. La marge est donc fine mais déterministe, pas chanceuse.

Elle a ensuite été vérifiée sur une **seconde graine de parties** (`validate_decision.py --seed 1`, 50 autres parties) : H0e reste
à 100 %, et H0f passe avec **plus** de marge — +4,02 points pour un minimum de 3, et 0,14 contact par partie pour un maximum de
0,5. Le détail est instructif : le planificateur est stable d'une graine à l'autre (score 1,10 puis 1,02 ; contacts 0,16 puis
0,14), et c'est la **référence** qui bouge, le bot aléatoire passant de −2,06 à −3,00. Autrement dit, la finesse apparente de la
marge sur la graine 0 venait d'un bot aléatoire anormalement chanceux, pas d'un planificateur limite. Cela confirme le choix de
traiter le nombre de contacts comme le critère décisif de H0f et le gain comme le critère secondaire.

**Révision du 15 novembre 2026, avant tout entraînement : réglages du niveau 2.** Le test de fumée a mesuré le débit
d'inférence sur T4 (715 images/s pour la régression, 242 pour la diffusion à n = 3) et en déduit le coût du niveau 2 aux
réglages par défaut : **1,0 h par run de régression mais 11,7 h par run de diffusion**, soit 38 h pour les six runs du cœur —
plus que la campagne d'entraînement elle-même, et quatre fois la réserve prévue en C5. Les jetons, à 66 h par run, confirment
par la mesure qu'ils doivent rester hors du niveau 2.

Le niveau 2 passe donc à **J = 32 suites et 20 parties** (au lieu de 64 et 50), identiques pour tous les runs comme l'exige la
comparabilité : le coût est divisé par cinq, soit ≈ 7,6 h au total. C'est le motif de révision explicitement prévu par
`EXTENSION_DECISION.md` (« réduire `--games` ou `--candidates` si la durée est trop longue, et le noter »), et il intervient
avant le premier entraînement. Contrepartie assumée : avec 20 parties au lieu de 50, l'erreur type sur les contacts par partie
passe de 0,06 à 0,09, ce qui rend H5a plus difficile à confirmer — dans le sens de la sévérité, donc acceptable.
**À faire avant le niveau 2** : relancer H0f à ces réglages (`validate_decision.py --games 20 --candidates 32`, sans GPU) pour
vérifier qu'un planificateur affaibli passe encore le contrôle. S'il échoue, garder J = 64 et ne réduire que les parties.

**Si H0f échoue chez vous.** Ne rien mesurer, et corriger le planificateur ou le lecteur avant toute chose : c'est la consigne
du dépôt, et elle ne change pas.

---

## C8 — La récompense est lue dans les pixels

**L'hypothèse concernée.** Tout le niveau 2 (H5a, H5b) : le planificateur lit la récompense dans les images que le modèle
imagine, au lieu de l'apprendre comme le fait PlaNet.

**Ce qui est en jeu.** C'est la limite principale du niveau 2. L'apprendre exigerait une tête de récompense et donc un
ré-entraînement de toutes les familles, hors budget.

**Décision.** La récompense reste lue dans les pixels, supposée connue. Le périmètre des conclusions est borné explicitement
dans le rapport.

**Pourquoi.** La récompense étant exacte, **tout écart observé au niveau 2 est imputable au seul modèle de dynamique** : c'est
une isolation plus propre de l'effet du flou qu'un système où l'erreur de récompense et l'erreur de dynamique se mélangent.
Ce que cela interdit, en revanche, c'est toute conclusion sur un système bout-en-bout : rien ici ne dit ce qui arriverait si la
récompense était elle aussi apprise sur des images floues. C'est une limite de portée, pas un défaut de mesure, et elle doit être
énoncée comme telle.

---

## Révision de ces décisions

Une décision de ce fichier peut être révisée **avant** le premier entraînement sans autre formalité que de modifier le texte.
Après le premier run, toute modification doit être ajoutée ici **à sa date**, avec son motif, sans effacer la décision
précédente : c'est la seule manière de distinguer une correction d'une hypothèse ajustée aux résultats. Un seul motif est
légitime après coup — le réglage mesure autre chose que ce qu'il annonce (règle 3) ; un seuil non atteint n'en est pas un.


### Révision du 15 novembre 2026, après le premier run : volume des données d'entraînement

**Ce qui était fixé.** Le protocole prévoyait 100 000 transitions d'entraînement par valeur de p
(`generate_data.py --n-transitions 100000`), chiffre hérité du dossier initial et jamais interrogé. Combiné aux 34 800 pas en
lot 64 fixés par C5, il donne 2 227 200 images vues pour 100 000 images distinctes, soit **22 passages** sur les données.

**Ce qui a été observé.** Le premier run réel, `core-regression-p0.30-s0`, échoue au contrôle H0b (rapport MSE/plancher
**1,47** contre la bande [1,00 ; 1,10]) et ne produit que 22,5 % de flou. Le diagnostic complet est dans
[`JOURNAL.md`](JOURNAL.md), entrée J6 : la perte d'entraînement finit à **0,46 × le plancher de Bayes** — infranchissable pour
un modèle qui ignore la branche, et il a été vérifié qu'il l'ignore — pendant que la perte de validation, après un plateau
jusqu'au pas 18 000, monte de façon monotone à partir du pas 20 000 pour finir **49 %** au-dessus de son minimum. Le modèle a mémorisé l'issue de chaque bifurcation vue à l'entraînement.

**Décision.** Le jeu d'entraînement passe à **400 000 transitions** pour les quatre valeurs de p, ce qui ramène le nombre de
passages de 22 à **5,6**. La validation (20 000 transitions) et le test (1 000 bifurcations) sont **inchangés** : leurs flux
aléatoires sont indépendants de celui de l'entraînement, donc rien n'est contaminé. Aucun pas, aucun lot, aucun seuil, aucune
hypothèse n'est modifié ; C5 reste à 34 800 pas et 28 h engagées.

**Motif, et sa légitimité.** C'est la règle 3, et elle seule : *on ne change un réglage que s'il mesure autre chose que ce
qu'on croit*. À 100 000 transitions, le dispositif mesurait la capacité du réseau à mémoriser un jeu de données, pas le flou
inhérent à la perte quadratique. Il faut en revanche assumer que ce changement va **dans le sens contraire de la règle 2** :
il rend H1 plus facile à confirmer, puisqu'il rétablit le flou attendu de la régression. Trois garde-fous l'encadrent. Le motif
est un **contrôle pré-enregistré qui a échoué**, pas un seuil qu'on trouvait trop sévère. Le changement s'applique
**identiquement aux quatre familles**, donc le budget apparié reste intact. Et le run fautif est **archivé, pas supprimé**
(`archive/core-regression-p0.30-s0_N100k/`), de sorte que la trajectoire complète reste vérifiable.

**Vérification exigée avant de relancer la campagne.** Un seul run aux nouvelles données (2,7 h) doit montrer une perte de
validation qui ne remonte plus jusqu'au pas 34 800, et un `ratio_expected` revenu dans [1,00 ; 1,10]. Tant que ces deux
conditions ne sont pas réunies, les 28 h ne sont pas engagées.

**À écrire dans le rapport.** Le premier run a été conduit à 100 000 transitions et a échoué à H0b par surapprentissage ; le
jeu d'entraînement a été porté à 400 000 avant toute mesure comparative, et aucun verdict n'est tiré du run initial.


### Révision du 16 novembre 2026, après le premier run : taille du jeu de test

**Ce qui était fixé.** 1 000 états de bifurcation par valeur de p, et une prédiction par état.

**Ce qui a été observé.** Une relecture externe du code a montré que **H2b ne pouvait structurellement pas être concluante** :
la règle exige au moins 1 000 images *nettes* par graine, or le maximum atteignable était exactement 1 000. Toute image floue
ou classée « autre » faisait tomber le verdict en « non concluant ». Le run archivé en comptait 2,0 % d'« autre ».

**Décision.** Le jeu de test passe à **2 000 états** par valeur de p. Le seuil `min_n_h2b = 1000` est inchangé : c'est une
exigence de précision (demi-largeur d'intervalle d'environ 0,028 sur p̂ à p = 0,3), pas un nombre arbitraire, et l'abaisser
aurait desserré une règle pré-enregistrée.

**Motif.** Règle 3 : le dispositif ne mesurait pas ce qu'il annonçait, puisqu'il rendait un verdict impossible à atteindre
indépendamment de la qualité des modèles. Ce n'est pas un seuil jugé trop sévère, c'est une impossibilité arithmétique.

**Conséquence à respecter.** ρmax est calibré sur les jeux de test : 2 000 états font 8 000 paires au lieu de 4 000, donc un
quantile 99,5 % différent. Relancer `validate_instrument.py`, revérifier H0a, et évaluer **tous** les runs avec le même ρmax.
Aucun entraînement n'est affecté : le jeu de test ne sert qu'à l'instrument et à l'évaluation.

**Au passage, C4 est enfin appliquée.** `analysis.py` appelait `compare_groups` sans `k_tests` et utilisait donc les seuils
k = 1. H2a et H3a reçoivent désormais k = 3, soit 4,416 s_d (apparié) et 3,234 s (non apparié). Ce n'est pas une révision mais
la mise en conformité du code avec une décision déjà prise. Réserve à écrire dans le rapport : H1 n'étant pas un test de
signification, k = 2 serait défendable (3,583 s_d) ; k = 3 est gardé comme choix conservateur.



### Révision du 17 novembre 2026, après cinq runs évalués : une alerte corrigée, deux mesures exploratoires déclarées

Trois décisions prises **avant** de lancer `analyze.py`, à la suite de l'incident J9
([`JOURNAL.md`](JOURNAL.md)). Aucune ne touche à un seuil ni à un verdict pré-enregistré : la première corrige un
instrument de surveillance, les deux autres ajoutent des mesures **déclarées exploratoires**, qui ne peuvent pas
confirmer une hypothèse.

**1. L'alerte de surapprentissage passe de « dernier point » à « tendance ».**

*Ce qui était fixé.* `scripts/training_curves.py` lève une alerte si la perte de validation finit plus de 10 % au-dessus
de son minimum.

*Ce qui a été observé.* La règle a signalé `core-diffusion-p0.30-s0` (+30,2 % sur le dernier relevé) alors que la
validation de ce run **descendait encore** : pente de log(validation) sur la seconde moitié de l'entraînement de −0,40
par 10 000 pas, t = −3,9. La validation de la diffusion fluctue fortement d'un relevé à l'autre même à bruit fixe, les
poids changeant entre deux relevés : +101 % au pas 14 000 pour ce même run, suivi d'une reprise de la baisse.

*Décision.* L'alerte se déclenche désormais quand la **pente de log(validation) sur la seconde moitié est positive avec
t > 3**. L'écart au minimum reste affiché, comme information. Sur les sept séries disponibles, la nouvelle règle signale
J6 (pente +0,21, t = +10,7) et lui seul.

*Motif.* Règle 3, appliquée à un instrument de surveillance et non à une hypothèse : la règle précédente mesurait le
bruit du dernier relevé, pas la mémorisation. Elle ne protège aucun verdict — elle décide seulement si l'on s'arrête
pour examiner un run.

*Réserve à écrire dans le rapport.* Le seuil t > 3 a été choisi **après** avoir vu ces sept runs, sur 9 relevés de
validation chacun (4 à 6 pour les stades des jetons). C'est un garde-fou, pas un test.

**2. Balayage de σ sur les débruiteurs déjà entraînés (exploratoire).**

*Ce qui a été observé.* La diffusion à n = 1 donne **100 % d'images « autre »** sur les deux graines, et une MSE à 2,1
fois le plancher de Bayes, là où le cours attendait la moyenne conditionnelle D(x ; σmax) — donc du flou, α̂ ≈ p et
ρ ≈ 0. Explication proposée : σmax = 80 n'a quasiment jamais été vu à l'entraînement (σ log-normale, P_mean = −1,2,
P_std = 1,2 : 4 exemples au-dessus de 80 et 525 au-dessus de 20, sur 2 227 200).

*Décision.* Mesurer, sans réentraîner, un pas d'Euler depuis σ ∈ {1, 2, 5, 10, 20, 80}
(`scripts/sigma_sweep.py`, sortie `runs/<run>/sigma_sweep.json`). **H2a reste évaluée telle que préenregistrée**, sur
n = 1 contre n = 3, et sera vraisemblablement réfutée. Le balayage sert à dire *pourquoi*, pas à changer le verdict.

*Motif.* Règle 1 : le seuil pré-enregistré est gardé bien qu'on sache déjà qu'il ne sera pas franchi. La règle 3 ne
s'applique pas, car ce n'est pas H2a qui mesure autre chose que ce qu'elle annonce — c'est l'échantillonneur à un pas
qui ne fait pas ce que le cours en attendait. Le diagnostic est donc rapporté à part.

*À écrire dans le rapport.* H2a réfutée, avec la couverture de σ à l'entraînement comme explication, et le balayage
comme pièce. Le fait que l'écart soit prévu *avant* de lire le verdict est ce qui distingue une explication d'une
justification après coup.

**3. Classifieur restreint au rail de la bifurcation (exploratoire, mesure secondaire).**

*Ce qui a été observé.* 51,8 % des images du run de jetons s0 sont classées « autre ». Elles sont nettes, mais l'agent
y est décalé d'un ou deux pixels. Sur les 25 images conservées par classe, 58 % de l'énergie de l'écart tombe sur
l'agent ; restreint aux pixels où A et B diffèrent, le classement donne 19 images nettes et 6 « autre ». La régression
montre le symétrique : ses rares « autre » viennent à 95 % du reste de l'image et deviennent « floue » au rail. C'est la
limite connue de l'instrument (cours, §2.2) : ρ est normalisé par δ, qui ne mesure que le déplacement de l'ennemi.

*Décision.* `evaluate.py` calcule désormais, **en plus** de la mesure principale et sur les 2 000 états, les mêmes
α̂, ρ et étiquettes restreints au rail (pixels où A et B diffèrent, dilatés de 2). Elles vivent sous la clé `rail` de
chaque `eval.json`, marquée `"exploratoire": true`. **Aucun verdict n'en dépend** : H0 à H5 se calculent sur les
étiquettes de l'image entière, inchangées.

*Motif.* Règle 3 pour la seule mesure secondaire : « autre » confond aujourd'hui deux défauts de nature différente, une
bifurcation ratée et un agent décalé. Le choix de **ne pas** remplacer la mesure principale relève de la règle 1 — le
classifieur sur l'image entière est ce qui a été pré-enregistré, et il reste ce sur quoi les hypothèses sont tranchées.

*Conséquence à respecter.* Les cinq runs déjà évalués l'ont été sans cette clé. Il faut les **réévaluer**
(`evaluate.py --force`, quelques minutes par run, aucun réentraînement) pour que les douze runs portent la même mesure,
sinon elle ne serait disponible que sur une partie de la campagne.


**Complément du 17 novembre, avant la session 3 : lecture de référence du balayage de σ, et une réserve sur la mesure au rail.**

Une relecture du commit `61dfcc6` (J10) a montré que la docstring de `sigma_sweep.py` prédisait « flou élevé » à σ = 1 à 5,
ce qui est **faux même pour un débruiteur exact**. La lecture du balayage est donc fixée ici, **avant** de voir ses chiffres.

*Le calcul.* L'échantillonneur d'Euler part de x = σ·n, du bruit pur. Pour un débruiteur exact sur {A, B}, le poids a
posteriori de A a un logit d'écart-type ‖A − B‖/σ, avec ‖A − B‖ ≈ 15,3 à l'échelle du modèle ([−1, 1]). Tant que σ < ‖A − B‖,
le bruit seul tranche la branche : l'image sort **nette**, avec ρ = 0. Le flou n'existe qu'aux grands σ. Référence calculée
par `scripts/ideal_sigma_reference.py` (NumPy seul), p = 0,3, 2 000 états, 40 tirages de bruit par état :

| σ | floue | nettes | α̂ p5 – p95 | autre |
|---|---|---|---|---|
| 1 | 7,1 % | 92,9 % | 0,000 – 1,000 | 0 |
| 2 | 14,4 % | 85,6 % | 0,000 – 1,000 | 0 |
| 5 | 33,8 % | 66,2 % | 0,003 – 0,985 | 0 |
| 10 | 56,4 % | 43,6 % | 0,034 – 0,842 | 0 |
| 20 | 75,9 % | 24,1 % | 0,109 – 0,600 | 0 |
| 30 | 85,6 % | 14,4 % | 0,158 – 0,496 | 0 |
| 40 | 92,3 % | 7,7 % | 0,187 – 0,444 | 0 |
| 60 | 98,4 % | 1,6 % | 0,221 – 0,394 | 0 |
| 80 | 99,8 % | 0,2 % | 0,239 – 0,370 | 0 |

*Comment la lire.* « Autre » vaut 0 à tout σ **par construction** : toute valeur mesurée au-dessus vient du réseau (grain,
point de départ). Ce qui sépare le réseau de cette référence, c'est le grain et la dispersion de α̂, pas la présence de
flou à σ petit.

*Prudence sur les σ petits.* Le réseau a été entraîné sur « image + σ·bruit », et les images cibles ont une moyenne de
−0,77 par pixel. Un bruit pur à σ = 1 s'en écarte de 85 écarts-types sur la moyenne, à σ = 10 de 8,5, à σ = 20 de 4, à
σ = 80 de 1. Le point de départ n'est dans la loi d'entraînement qu'à σ ≳ 40, où il y a 51 exemples (σ ≥ 40) ou 4 (σ ≥ 80),
contre 21 400 à σ ≥ 5. **Un mauvais résultat à σ ≤ 10 ne dirait donc pas « σ mal couvert »** : il peut venir du point de
départ. C'est pourquoi la plage va jusqu'à 80 avec des points intermédiaires (1, 2, 5, 10, 20, 30, 40, 60, 80).

*Observation à regarder, sans en faire une conclusion.* Le α̂ mesuré à n = 1 va de 0,03 à 0,86 (5e–95e centiles, J9), ce que
le débruiteur exact donne à σ ≈ 10 (0,034 – 0,842) et non à σ = 80 (0,239 – 0,370). Le réseau se comporterait à σ = 80 comme
s'il voyait un bruit dix fois plus faible. Rapprochement non vérifié sur le réseau.

*Réserve sur la décision 3 (mesure au rail).* La mesure au rail réutilise ρmax = 0,1599, calibré sur l'image entière. Au
rail, le même bruit synthétique (σ = 0,01, quantile 99,5 %) donne un ρ de **0,036** au lieu de 0,164, mesuré sur 300 états de
p = 0,3 : le rail est 4,6 fois plus tolérant que la mesure principale, ses « autre » sont donc **un minorant**. Rien n'est
recalibré, pour que la mesure reste exploratoire et comparable aux chiffres de J9 ; mais `eval_arrays.npz` garde `__rho_rail`
par état, de sorte qu'on peut re-seuiller après coup, sans réévaluer. À écrire dans le rapport.
