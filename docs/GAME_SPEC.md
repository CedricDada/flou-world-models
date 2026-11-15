# Spécification du mini-jeu

Le jeu est l'**instrument de mesure** du projet : un monde minuscule dont on connaît exactement la
loi de transition. Tout le hasard est concentré en un seul endroit, la bifurcation de l'ennemi,
et sa probabilité `p` est un paramètre. Ce document décrit les règles telles qu'elles sont codées
(`src/flou/game/`) ; en cas de doute, le code et les tests font foi.

## 1. Coordonnées et objets

Image de 64 × 64 pixels RGB. `x` croît vers la droite, `y` vers le bas. Un objet carré de côté `s`
en `(x, y)` occupe les pixels `[x, x+s) × [y, y+s)`. Deux objets se **chevauchent** si leurs
boîtes ont une intersection non vide ; le simple contact ne compte pas.

| Objet | Taille | Couleur (RGB) | Rôle |
|---|---|---|---|
| Murs d'enceinte | 2 px d'épaisseur | (80, 80, 95) | délimitent l'arène `[2, 62)²` |
| 4 piliers | 6 × 6, en (12,12) (46,12) (12,46) (46,46) | (80, 80, 95) | murs intérieurs : arrêtent l'agent, font rebondir l'obstacle |
| Agent | 4 × 4 | (50, 130, 255) | piloté par les 5 actions |
| Obstacle mobile | 6 × 6 | (200, 200, 210) | rebondit de façon déterministe |
| Ennemi | 6 × 6 | (235, 55, 55) | va et vient sur un rail ; bifurque au centre |
| Petit objet | 2 × 2 | (255, 215, 40) | à ramasser ; sert au « rappel » (H3) |
| Fond | | (12, 12, 20) | |

Ordre de superposition, du fond vers le dessus : murs, obstacle, ennemi, agent, petit objet. Le
petit objet est dessiné en dernier : sa couleur est unique et il n'est jamais masqué, donc sa
détection est non ambiguë.

## 2. L'état exact

Dix entiers par partie (`flou.game.State`, ordre des colonnes de `to_array()`) :

| Colonne | Champ | Signification |
|---|---|---|
| 0, 1 | `ax`, `ay` | agent : coin haut-gauche |
| 2, 3 | `ox`, `oy` | obstacle : coin haut-gauche |
| 4, 5 | `ovx`, `ovy` | obstacle : vitesse en pixels par pas, de valeur absolue 1 ou 2, jamais nulle |
| 6 | `eo` | ennemi : décalage sur le rail en nombre de pas, de −3 à +3 (0 = centre) |
| 7 | `ev` | ennemi : sens de déplacement, +1 (droite) ou −1 (gauche) |
| 8, 9 | `gx`, `gy` | petit objet : coin haut-gauche |

L'état ne contient **pas** la branche : c'est un argument de `advance`, pas une partie du monde.
Ni `ev` ni les vitesses de l'obstacle ne se lisent sur une seule image ; elles se déduisent de
deux images consécutives, d'où le contexte de plusieurs images donné aux modèles.

## 3. Un pas de jeu : `advance(state, action, branch, cfg)`

Fonction pure et vectorisée : elle renvoie `(état suivant, événements)`. Ordre de mise à jour :

1. **Agent.** Actions : 0 ne rien faire, 1 haut, 2 bas, 3 gauche, 4 droite ; déplacement de
   2 pixels. Le mouvement est refusé (drapeau `blocked`) s'il fait entrer l'agent dans un mur ou
   un pilier, ou s'il **crée** un chevauchement avec l'obstacle. Un agent déjà chevauché par
   l'obstacle peut donc en sortir.
2. **Obstacle.** Il avance de `(ovx, ovy)`. Axe x d'abord : si la nouvelle boîte touche un mur,
   il reste en place sur cet axe et `ovx` change de signe (drapeau `bounce_x`) ; puis axe y de la
   même façon (`bounce_y`). Aucun hasard.
3. **Ennemi.** Le rail est horizontal, centré, de 3 pas de part et d'autre du centre (18 px).
   - Au centre (`eo == 0`), la **branche** décide : **A → vers la droite**, **B → vers la
     gauche**. Les deux positions possibles sont à 12 px l'une de l'autre (le cours préparatoire
     propose au moins 8 px, voir `MIN_BRANCH_DIST`).
   - Ailleurs il continue dans le même sens ; au bout du bras (`|eo| == 3`) il fait demi-tour ;
     de retour au centre, il rebifurque. Un cycle dure 6 pas : un état sur six est un état de
     bifurcation (16,7 %), quel que soit le bot.
4. **Événements.** `hit_enemy` et `hit_obstacle` (l'agent chevauche l'objet après le pas),
   `pickup` (l'agent chevauche le petit objet), `branch` (branche utilisée : −1, 0 pour B ou 1
   pour A).
5. **Petit objet.** Après un ramassage, il réapparaît à celui des 8 emplacements fixes
   (4 coins, 4 milieux de côté) qui est **le plus éloigné de l'agent** (égalité : plus petit
   indice). La règle ne dépend que de ce qui est visible : elle n'ajoute aucun hasard caché.

L'ennemi n'agit pas sur le reste du monde : toucher l'ennemi ne lève qu'un drapeau (et un
malus de −1 dans la récompense). Ainsi, **aux états de bifurcation, les deux futurs ne diffèrent
que par la position de l'ennemi** (test `test_branch_changes_only_the_enemy`).

Récompense (`VecMiniGame.step`) : +1 par ramassage, −1 par contact avec l'ennemi. Elle ne sert
pas aux modèles du monde ; elle sert aux bots et à d'éventuelles extensions.

## 4. Le hasard

- `VecMiniGame.step` tire `u ~ U[0,1)` à chaque pas, pour chaque partie (le flux aléatoire est
  donc le même quel que soit le moment des bifurcations), et la branche vaut A si `u < p` à
  un état de bifurcation. `p` est le champ `GameConfig.p`.
- La branche est tirée **à l'instant t** et n'apparaît dans l'image qu'en **t + 1**. L'image
  de `s_t` ne dépend que de `s_t`. Les tests vérifient que la fréquence de A vaut `p` et que le
  tirage est indépendant du précédent.
- L'état initial est tiré au hasard (agent, obstacle et sa vitesse, phase de l'ennemi, petit
  objet) sans chevauchement agent/autres objets ; toutes les transitions suivantes sont
  déterministes à la branche près.

## 5. Rendu contrefactuel

`render_counterfactual(état, action, cfg)` renvoie les images `A` et `B` du pas suivant pour un
même état et une même action. C'est ce qui donne la « vérité » pour la sonde du classifieur de
flou : une prédiction `ŷ` se projette sur le segment `[A, B]`.

- `δ = ||A − B||` (norme euclidienne, pixels ramenés à [0, 1]) vaut `√(2·36·||c_ennemi − c_fond||²) ≈ 7,65`
  quand l'ennemi ne recouvre rien d'autre ; un peu moins s'il passe sous l'obstacle ou sur l'agent.
- Hors d'un état de bifurcation, `A == B` et `δ = 0`.
- Le mélange `p·A + (1−p)·B` est exactement ce que prédit une régression MSE idéale ;
  `docs/img/contrefactuel.png` le montre.

## 6. Ce que les bots lisent et ne lisent pas

Les bots reçoivent l'état exact. Ils ne tirent **jamais** la branche et ne la connaissent pas à
l'avance : l'expert anticipe les deux branches et retient le pire cas.

## 7. Limites connues (à garder en tête pour l'interprétation)

- **Le lieu et le rythme de la bifurcation sont prévisibles** (centre du rail, tous les 6 pas) ;
  seule l'issue est imprévisible. C'est voulu : un modèle sait qu'une bifurcation arrive et doit
  décider quoi prédire. En revanche, le projet ne dit rien sur la prédiction *du moment* d'une
  bifurcation.
- **Une seule source de hasard**, un ennemi qui n'interagit pas avec le reste. Conséquence : le
  jeu isole le phénomène étudié mais ne dit rien de dynamiques où la bifurcation modifie la
  suite du jeu.
- L'ennemi va 3 fois plus vite que l'agent (6 px contre 2 par pas).
- Les valeurs numériques (tailles, vitesses, longueur du rail) sont des **choix de départ**,
  modifiables dans `GameConfig` ; la validation de la configuration refuse celles qui violent
  les contraintes retenues (distance entre branches ≥ 8 px, rail dans l'arène, etc.).
