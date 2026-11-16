# Pièces du premier run : `core-regression-p0.30-s0` à 100 000 transitions

Mesures brutes du run qui a révélé le surapprentissage de la régression, conservées telles qu'écrites par le code. Le récit et
l'analyse sont dans [`../../JOURNAL.md`](../../JOURNAL.md) (entrée J6) ; la décision qui en découle est datée à la fin de
[`../../DECISIONS.md`](../../DECISIONS.md).

**Ne pas lire ces chiffres comme un résultat du projet.** Ils proviennent d'une configuration abandonnée (100 000 transitions
d'entraînement, remplacées par 400 000) et aucun verdict n'en est tiré. Ils sont là comme pièce à conviction.

| Fichier | Contenu |
|---|---|
| `log.jsonl` | 366 lignes : 348 journaux d'entraînement (tous les 100 pas) et 18 de validation (tous les 2 000 pas) |
| `eval.json` | Évaluation sur les 1 000 états de bifurcation du jeu de test : comptages, H0b, rollouts |

Conditions : Tesla T4, PyTorch 2.14.1+cu130, 34 800 pas en lot 64, p = 0,3, graine 0, 2,69 h.

## Les deux chiffres qui comptent

**La validation remonte.** Plateau bruité du pas 4 000 au pas 18 000 (minimum 0,000800 au pas 8 000), puis hausse monotone à
partir du pas 20 000 jusqu'à 0,001195, soit **+49 %** au-dessus du minimum, pendant que la perte d'entraînement continue de
descendre. En passages sur les données : plateau de 2,6 à 11,5, hausse soutenue à partir de 12,8.

**La perte d'entraînement passe sous le plancher de Bayes.** Elle finit à 0,000305 pour un plancher de 0,000659 — soit
**0,46 ×**. Cette borne est infranchissable pour un modèle qui ignore quelle branche a été tirée ; il a été vérifié que la
régression l'ignore (la branche n'entre dans le réseau que si `oracle=True`). Le modèle a donc mémorisé l'issue de chaque
bifurcation vue à l'entraînement.

Le plancher se recalcule depuis les données : part de transitions de bifurcation 16,7 %, δ² moyen 57,9, p(1−p) = 0,21, converti
dans les unités de `mse_loss` (moyenne par élément, échelle −1…1).

## Relire les courbes

```python
import json
rows = [json.loads(l) for l in open('log.jsonl', encoding='utf-8')]
train = [(r['step'], r['loss']) for r in rows if 'loss' in r]
val = [(r['step'], r['val_loss']) for r in rows if 'val_loss' in r]
```
