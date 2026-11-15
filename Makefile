# Raccourcis. Avec uv (recommandé), chaque cible tourne dans l'environnement .venv du projet, créé et mis à jour
# automatiquement par `uv run`. Sans uv : `make test PY=python` (les dépendances doivent déjà être installées).
#   PY  : interpréteur des cibles qui n'ont pas besoin de PyTorch
#   PYT : interpréteur des cibles qui en ont besoin (avec uv : ajoute l'extra `train`)
ifeq ($(shell command -v uv 2>/dev/null),)
PY  ?= python
PYT ?= python
else
PY  ?= uv run python
PYT ?= uv run --extra train python
endif

# Sans uv, ou avec un environnement non installé : PYTHONPATH évite d'installer le paquet.
export PYTHONPATH := src

.PHONY: help sync sync-train test lint play bench demo data-train data-val data-test data plan validate validate-decision dry-run analyze smoke decision clean

help:
	@echo "Installation : make sync (tout sauf PyTorch) ; make sync-train (PyTorch en plus, machine d'entraînement)"
	@echo "Vérification : make test, make lint"
	@echo "Jeu          : make play, make demo, make bench"
	@echo "Données      : make data"
	@echo "Sans GPU     : make validate, make validate-decision, make plan, make dry-run, make analyze"
	@echo "Avec GPU     : make smoke (une fois), puis scripts/run_plan.py (voir README §3), puis make decision"
	@echo "Interpréteur : $(PY)   (changer : make test PY=python)"

# --- Environnement (uv) ---
sync:
	uv sync

# Machine d'entraînement : `--extra train` est indispensable, car `uv sync` seul désinstallerait PyTorch.
sync-train:
	uv sync --extra train

# --- Vérifications ---
test:
	$(PY) -m pytest

lint:
	$(PY) scripts/lint_names.py src scripts tests

# --- Jeu ---
play:
	$(PY) scripts/play.py

bench:
	$(PY) scripts/benchmark.py

demo:
	$(PY) scripts/demo.py --out docs/img

# --- Données ---
# p = 0 : contrôles (régression et diffusion sans bifurcation) ; 0,1 / 0,3 / 0,5 : plan principal et axe p
data-train:
	$(PY) scripts/generate_data.py --split train --p 0.0 0.1 0.3 0.5 --n-transitions 100000

data-val:
	$(PY) scripts/generate_data.py --split val --p 0.0 0.1 0.3 0.5 --n-transitions 20000

data-test:
	$(PY) scripts/generate_data.py --split test --p 0.0 0.1 0.3 0.5 --n-bifurcations 1000

data: data-train data-val data-test

# --- Instruments et plan (sans GPU) ---
validate:
	$(PY) scripts/validate_instrument.py

# Extension « décision » (docs/EXTENSION_DECISION.md) : contrôles H0e et H0f, ~5 min sur 2 cœurs
validate-decision:
	$(PY) scripts/validate_decision.py

plan:
	$(PY) scripts/plan.py

dry-run:
	$(PY) scripts/dry_run.py

analyze:
	$(PY) scripts/analyze.py

# --- À lancer une fois sur la machine GPU avant tout entraînement (PyTorch requis) ---
smoke:
	$(PYT) scripts/smoke_test.py --time

# Extension « décision », niveau 1 (H4) sur les runs entraînés du cœur ; le niveau 2 (H5, coûteux) se lance à la main :
#   uv run --extra train scripts/evaluate_decision.py --all --level 2
decision:
	$(PYT) scripts/evaluate_decision.py --all

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
