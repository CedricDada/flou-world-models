"""Permet de lancer les scripts sans installer le paquet : `python scripts/xxx.py`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TORCH_HELP = ("PyTorch est requis pour cette étape (modèles, entraînement, évaluation d'un run) et n'est pas installé.\n"
              "  avec uv  : uv sync --extra train        (puis : uv run scripts/...)\n"
              "             (ne pas relancer un `uv sync` nu ensuite : il désinstallerait PyTorch)\n"
              "  avec pip : pip install torch            (voir https://pytorch.org pour la version GPU)\n"
              "  sur Colab : déjà installé.\n"
              "Les autres étapes (jeu, données, instruments, analyse, répétition générale) n'en ont pas besoin.")


def have_torch() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def require_torch():
    """Importe PyTorch ou s'arrête avec un message qui dit quoi faire (au lieu d'une trace d'erreur)."""
    try:
        import torch
    except ImportError:
        raise SystemExit(TORCH_HELP) from None
    return torch
