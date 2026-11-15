# Extension « décision » (niveaux 1 et 2) : hypothèses H4 et H5

**Statut.** Le code est écrit. La partie NumPy (lecteur de contact, coût exact, planificateur, contrôles H0e et H0f, analyse,
répétition générale) est **exécutée et testée**. La mesure des modèles entraînés (`scripts/evaluate_decision.py`,
`flou.train.evaluation`) est en PyTorch : **écrite, jamais exécutée**, comme le reste du code PyTorch. **Aucun modèle n'a été
mesuré : H4 et H5 n'ont pas de résultat.** Les hypothèses et leurs seuils ont été écrits avant toute mesure d'un modèle ; les seuils
marqués **[choix]** sont des valeurs de départ, à confirmer avant les premiers runs (`TODO.md`, C6 à C8).

**Règle de citation.** Chaque choix de conception de cette extension (ici, dans le code, dans le README) indique l'article qui l'a
guidé, ou dit explicitement qu'il vient de notre propre raisonnement. Les articles ont été lus par extraction ciblée de leurs PDF
(questions précises sur les sections utiles), **pas en entier** ; les chiffres cités viennent de ces extractions.

## 1. Pourquoi

Le dépôt mesure si une prédiction est nette ou floue. Il ne dit pas si cela **change une décision**. Le niveau 1 mesure l'effet sur
l'estimation d'un risque (contact avec l'ennemi). Le niveau 2 mesure l'effet sur le score d'un agent qui **planifie avec le modèle
comme simulateur** : c'est l'usage qui justifie le mot « world model ». Le niveau 3 (agent entraîné par apprentissage par renforcement
dans le modèle) reste hors budget : DIAMOND annonce ≈ 2,9 jours sur une RTX 4090 par partie et par graine.

## 2. Sources qui nous guident

| Article | Ce que nous en retenons | Où ça sert |
|---|---|---|
| **PETS**, Chua, Calandra, McAllister, Levine, 2018, arXiv:1805.12114 | Contrôle prédictif (MPC) : on exécute seulement la première action de la suite optimisée, puis on replanifie (§6, Alg. 1). Optimiseur CEM (§5, annexe A.8). Propager l'**espérance** de la prédiction avec une seule particule est une base de comparaison (§5.2) ; les auteurs avertissent que de petits biais du modèle s'y accumulent vite. Classement des modèles : probabiliste ensembliste > probabiliste > déterministe ensembliste > déterministe ; les méthodes de propagation avancées n'apportent que des gains mineurs, le choix du modèle compte davantage (§7.2). Leurs expériences utilisent 20 particules (§5.1). | boucle du planificateur ; H4a, H5a ; la régression joue le rôle de la propagation de l'espérance, la diffusion et les jetons celui de l'échantillonnage de trajectoires |
| **PlaNet**, Hafner et al., 2019, arXiv:1811.04551 | Planification en ligne par CEM, replanification à chaque pas (§2) ; réglages : horizon 12, 10 itérations, 1 000 suites candidates, 100 élites (annexe A) ; **une seule trajectoire par suite candidate suffit** (§2) ; la planification somme des récompenses prédites **sans générer d'images** ; un planificateur avec le vrai simulateur sert de référence (Table 1). | forme du planificateur ; contrôle H0f ; justification du budget réduit (nous générons des images) |
| **Objective Mismatch**, Lambert, Amos, Yadan, Calandra, 2020, arXiv:2002.04523 | La précision du modèle (vraisemblance à un pas) et la qualité du contrôle sont des objectifs distincts. Corrélations mesurées entre vraisemblance et récompense : de −0,06 à 0,59 selon l'environnement et le jeu de données, sur 1 000 modèles (cartpole) et 2 400 (half-cheetah) (§4.1, Fig. 3) ; pas de garantie qu'améliorer la précision améliore le contrôle. | H5b ; prudence sur l'effet attendu : H4 et H5 peuvent être réfutées |
| *Contexte, non lus en détail* : DIAMOND (Alonso et al., 2024, arXiv:2405.12399), IRIS (Micheli et al., 2022, arXiv:2209.00588) | Agents entraînés dans un modèle du monde ; coût de DIAMOND (extrait lu via une recherche). | seulement l'argument de coût du niveau 3 |

**Non issu d'un article (notre raisonnement, que H4 teste).** Un lecteur d'image *affine* appliqué à l'image moyenne
`p·A + (1−p)·B` rend exactement le bon risque à un pas : le flou n'est donc pas une faute pour un tel lecteur. Il ne devrait nuire
qu'à un lecteur à seuil, ou quand le modèle reçoit en entrée ses propres images floues (il n'en a jamais vu à l'entraînement).

## 3. Définitions, et ce qui a été mesuré sans aucun modèle

Tout ce qui suit est mesuré sur le **vrai jeu** (NumPy, p = 0,3) avec `uv run scripts/validate_decision.py` ; les chiffres viennent
d'une seule graine : ce sont des ordres de grandeur.

**Coût de contact à l'horizon k** : nombre de pas, parmi les k suivants, où l'agent chevauche l'ennemi (à k = 1 : la probabilité de
contact). C'est ce que compte la récompense du jeu (−1 par pas de contact) et c'est une quantité linéaire :
`C* = p·C_A + (1−p)·C_B`. Après un état de bifurcation, le suivant n'arrive que 6 pas plus tard : pour **k ≤ 5** un seul tirage de
branche compte, `C*` est **exact** (vérifié par un test).

**États à risque ambigu** (`C_A ≠ C_B`) : 8,8 % des 3 000 états de test à k = 1, 14,0 % à k = 3, 18,4 % à k = 5. On mesure tous ces
états, plus autant d'états non ambigus tirés au hasard (pour voir les fausses alertes).

**Lecteur de contact** (`flou.eval.decision.ContactReader`). L'agent est localisé par sa couleur ; l'ennemi ne vit que sur les
7 positions de son rail ; pour chacune on mesure la part de « rouge » (R − (G+B)/2, ramené à 0 pour le fond et 1 pour l'ennemi) sur
les pixels que l'agent ne couvre pas. **Lecteur doux** : somme de ces parts sur les positions que la boîte de l'agent chevauche, dans
[0, 1] ; il est affine en l'image. **Lecteur à seuil** : le même, seuillé à 0,5 (contact : oui ou non).
*Contrôle H0e, mesuré :* sur 30 000 vraies images (887 contacts), le lecteur retrouve l'événement du jeu dans **100 %** des cas.

**Erreur** `E_k` = moyenne de |Ĉ − C*| sur les états à risque ambigu. Régression : un rollout déterministe. Générateurs : moyenne de
m = 8 rollouts. Valeurs obtenues en faisant passer des prédicteurs **factices** (dont on connaît le comportement) dans la chaîne :

| Prédicteur factice | doux k=1 | seuil k=1 | doux k=3 | seuil k=3 | doux k=5 | seuil k=5 | théorie k=1 |
|---|---|---|---|---|---|---|---|
| régression idéale (image moyenne exacte) | 0,001 | 0,300 | 0,001 | 0,345 | 0,002 | 0,447 | 0 / 0,300 |
| générateur net calibré, m = 8 | 0,132 | 0,132 | 0,157 | 0,157 | 0,206 | 0,206 | 0,133 |
| générateur net avec le biais d'Euler du cas jouet (p̂ = 0,188), m = 8 | 0,151 | 0,151 | 0,175 | 0,175 | 0,224 | 0,224 | 0,152 |

Lecture : avec le lecteur doux, une régression idéale estime le risque **mieux** qu'un générateur parfait à 8 tirages (le flou
contient exactement la probabilité ; le générateur paie le bruit de ses tirages). Avec le lecteur à seuil c'est l'inverse.

**Planificateur** (`flou.eval.planning`). À chaque pas du vrai jeu : J = 64 suites uniformes de H = 8 actions ; le modèle imagine les
images (m = 1 tirage pour la régression, m = 4 pour les générateurs) ; retour d'une suite = +1 si l'agent imaginé atteint le petit
objet réel, −1 par contact lu (lecteur doux par défaut), −λ·distance(agent imaginé, petit objet)/64 en fin d'horizon si le petit
objet n'est pas atteint (λ = 0,5) ; on joue **la première action de la meilleure suite** et on recommence. Tous les modèles jouent
les **mêmes 50 parties** de 100 pas. Mesures avec des « modèles » factices tirés du vrai jeu (50 parties, mêmes graines) :

| Joueur | Score | Contacts / partie | Ramassages / partie |
|---|---|---|---|
| bot aléatoire | −2,06 | 2,08 | 0,02 |
| bot expert (lit l'état exact) | 1,66 | 0,04 | 1,70 |
| **planificateur, vrai jeu, m = 4 (contrôle H0f)** | **1,10** | **0,16** | 1,26 |
| planificateur, vrai jeu avec le biais d'Euler (p̂ = 0,188), m = 4 | 1,16 | 0,14 | 1,30 |
| planificateur, vrai jeu, m = 1 | 0,88 | 0,36 | 1,24 |
| planificateur, image moyenne exacte, lecteur doux | 1,04 | 0,02 | 1,06 |
| planificateur, image moyenne exacte, lecteur à seuil | 0,56 | 0,60 | 1,16 |

Trois enseignements, **avant tout entraînement** :

1. Le contrôle H0f passe, mais de peu sur le critère de gain (+3,16 points pour un seuil de 3 ; le score du bot aléatoire varie de
   −2 à −3,3 selon les parties tirées).
2. Sans le terme de rapprochement (λ = 0), le planificateur évite l'ennemi mais ne ramasse presque rien (0,19 ramassage par
   partie sur 32 parties, score 0,09) : le petit objet est trop loin pour 8 pas. D'où ce terme, qui est notre ajout.
3. **Avec des modèles parfaits, H5a serait réfutée.** Une image moyenne *exacte*, lue par le lecteur doux, donne le meilleur
   évitement (0,02 contact par partie contre 0,16 pour le vrai jeu échantillonné 4 fois). H5a ne peut donc être confirmée que si
   la régression **réelle** se dégrade quand elle reçoit ses propres images floues en entrée : c'est précisément l'accumulation des
   biais que signale PETS (§5.2), et c'est ce que l'expérience dira.

## 4. Niveau 1 : le modèle estime-t-il bien un risque ?

| | Énoncé | Confirmée si | Réfutée si | Guidé par |
|---|---|---|---|---|
| **H0e** (contrôle) | Le lecteur appliqué aux **vraies** images (branches A et B, k = 1 à 5) retrouve l'événement de contact du jeu. | accord ≥ 99 % (mesuré : 100 %) | accord < 99 % : on ne mesure aucun modèle | notre règle H0a (instrument validé avant mesure) |
| **H4a** | Lecteur à seuil, k = 1 : la régression estime moins bien le risque que la diffusion à n = 3. | E(régression) − E(diffusion) ≥ 0,10 **[choix]** et > 2,27 écarts-types poolés (3 graines contre 3) | écart < 0,10 | PETS §5.2, §7.2 ; écart attendu avec des modèles parfaits : 0,15 à 0,17 |
| **H4b** | Lecteur doux, k = 1 : la régression estime le risque **au moins aussi bien** que la diffusion à n = 3. | E(régression) ≤ E(diffusion) + 0,02 **[choix]** | E(régression) > E(diffusion) + 0,05 ; entre les deux : non concluante | notre raisonnement de linéarité (§2) ; PETS §7.2 (gains mineurs de la propagation) |
| **H4c** (exploratoire) | `E_k` pour k = 1, 3, 5, par famille et par lecteur : l'erreur de la régression croît-elle plus vite que celle des générateurs ? | sans seuil, table et courbes | — | PETS §5.2 (accumulation des biais) |

## 5. Niveau 2 : le flou coûte-t-il des points à un agent qui planifie ?

| | Énoncé | Confirmée si | Réfutée si | Guidé par |
|---|---|---|---|---|
| **H0f** (contrôle) | Avec le **vrai jeu** comme modèle, le planificateur fonctionne. | ≤ 0,5 contact par partie et ≥ 3 points de plus que le bot aléatoire sur les mêmes parties **[choix]** (mesuré : 0,16 ; +3,16) | sinon : planificateur ou lecteur à corriger, rien n'est conclu | PlaNet, Table 1 (référence avec le vrai simulateur) |
| **H5a** | Le planificateur à modèle **flou** (régression) subit plus de contacts que celui à modèle **net** (diffusion n = 3). | contacts(régression) − contacts(diffusion) ≥ 0,5 par partie **[choix]** et > 2,27 écarts-types poolés | écart < 0,5 | PETS §5.2, §7.2 ; prudence de Lambert et al. §4.1 ; voir §3, point 3 |
| **H5b** (exploratoire) | Corrélation de rang entre le score du planificateur et : part de prédictions nettes, MSE par pixel, suivi à 16 pas. | sans seuil, 6 à 9 runs du cœur : indicatif seulement | — | Lambert et al. §4.1, Fig. 3 |

Choix du planificateur et leur origine : exécuter la première action puis replanifier (PETS §6, Alg. 1 ; PlaNet §2) ; m tirages par
suite pour les générateurs (particules de PETS §5.1) et un seul pour la régression (propagation de l'espérance, PETS §5.2) ; tirage
simple de 64 suites au lieu d'un CEM à 10 × 1 000 suites, parce que PlaNet n'atteint ce régime qu'en planifiant sans images
(annexe A) ; récompense **lue dans les pixels**, donc supposée connue, alors que PlaNet l'apprend (§2) : c'est une limite assumée ;
lecteur doux par défaut : le plus favorable à la régression, donc le test le plus sévère pour H5a (notre choix).

**Limites écrites d'avance.** 3 graines contre 3 : puissance d'environ 46 % pour un effet de 2 écarts-types
(`docs/EXPERIMENTS.md` §4) ; un résultat sous le seuil est « non concluant », jamais « pas d'effet ». Une réfutation de H4 ou de H5
serait un résultat, pas un échec : elle irait dans le sens du désaccord entre précision et contrôle documenté par Lambert et al.
Un seul planificateur, une seule valeur de p (0,3), une récompense connue : rien ne dit que les conclusions tiennent ailleurs.

## 6. Mode d'emploi

```bash
uv run scripts/validate_decision.py        # sans GPU, ~5 à 10 min : jeu de test du niveau 1, H0e, table des attendus, H0f
uv run scripts/dry_run.py                  # sans GPU : la répétition générale inclut H4 et H5 (résultats SIMULÉS)
# après les entraînements (GPU) :
uv run scripts/evaluate_decision.py --all                                      # niveau 1 -> runs/<run>/decision.json
uv run scripts/evaluate_decision.py --all --level 2 --games 4 --game-steps 10 --no-save   # chronométrer le niveau 2
uv run scripts/evaluate_decision.py --all --level 2                            # niveau 2 -> runs/<run>/planning.json
uv run scripts/analyze.py                  # verdicts H0e, H0f, H4a-c, H5a-b ajoutés au rapport
```

| Fichier | Rôle | État |
|---|---|---|
| `src/flou/eval/decision.py` | lecteur de contact, coût exact, erreur `E_k`, prédicteurs factices de rollout | testé |
| `src/flou/eval/planning.py` | planificateur, parties dans le vrai jeu, simulateurs factices, contrôle H0f | testé |
| `src/flou/eval/decision_instrument.py`, `scripts/validate_decision.py` | H0e, table des attendus, H0f | testé |
| `src/flou/eval/decision_analysis.py` | verdicts H4 et H5, tables, section du rapport | testé (runs simulés) |
| `src/flou/train/evaluation.py` (`evaluate_decision_run`, `plan_run`), `scripts/evaluate_decision.py` | mesure des modèles entraînés | **jamais exécuté** |
| `tests/test_decision.py`, `tests/test_planning.py` | 49 tests | passent |

**Coût.** Aucun ré-entraînement. Niveau 1 : quelques centaines d'états × 5 pas × 8 tirages, négligeable. Niveau 2 : 50 parties ×
100 décisions × 64 suites × 8 pas = 2,6 millions d'images par run pour la régression, 10,2 millions pour la diffusion (× 3 passes
du réseau à n = 3). **Durée non mesurée** : `smoke_test.py --time` mesure le débit d'inférence et affiche une estimation ; elle
s'ajoute aux ≈ 38 h du plan. Les jetons sont exclus du niveau 2 par défaut (64 passes du Transformer par image).

**Dans la répétition générale**, l'effet planté côté régression pour le niveau 2 (l'ennemi disparaît des images imaginées) est
**inventé** et volontairement énorme : il sert à vérifier que l'analyse sait confirmer H5a, pas à prédire quoi que ce soit.

## 7. Décisions à confirmer avant les premières mesures

**Toutes tranchées le 15 novembre 2026 — voir [`DECISIONS.md`](DECISIONS.md), C6 à C8 :** H5a gardée telle quelle avec le lecteur
doux, seuils et réglages du planificateur confirmés, récompense lue dans les pixels assumée. Ce qui suit est la liste d'origine.

1. Garder H5a telle quelle (en sachant qu'elle est réfutée pour des modèles parfaits, §3), ou la reformuler.
2. Seuils : 0,10 (H4a), 0,02 et 0,05 (H4b), 0,5 contact par partie (H5a), contrôle H0f (qui passe de peu).
3. Planificateur : J = 64, H = 8, m = 4, λ = 0,5, 50 parties, lecteur doux, p = 0,3 seulement.
4. Jeu de test du niveau 1 : 3 000 états (≈ 260 états ambigus à k = 1), distinct du jeu de test principal.
5. Récompense lue dans les pixels (supposée connue) plutôt qu'apprise : l'apprendre demanderait de ré-entraîner.
