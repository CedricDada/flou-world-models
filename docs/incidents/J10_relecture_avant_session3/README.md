# Pièces de J10 : relecture du commit 61dfcc6 avant la session 3

Note de l'agent externe, conservée telle qu'il l'a rendue. Le récit, la vérification indépendante de chaque point et
ce qui en a été appliqué sont dans [`../../JOURNAL.md`](../../JOURNAL.md) (entrée J10) ; la lecture de référence du
balayage de σ est dans [`../../DECISIONS.md`](../../DECISIONS.md) (complément du 17 novembre).

**Écarts entre la note et ce qui a été appliqué.** La note propose de lancer le balayage depuis le notebook sans changer
le script ; le script a aussi été modifié (docstring corrigée, plage par défaut, idempotence). Elle propose une variante
idempotente de la réévaluation par `subprocess` ; le notebook utilise `!python` pour voir la sortie en direct, avec un
`try` autour. Les chiffres de la note (quantile de ρ au rail 0,036, table de référence, comptes de σ) ont été recalculés
et se reproduisent.
