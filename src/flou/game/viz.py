"""Aides d'affichage (Pillow, facultatif) : planches d'images agrandies."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _pil():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow est requis pour les images : `uv sync` (groupe viz, installé par défaut) ou `pip install pillow`") from exc
    return Image, ImageDraw


def contact_sheet(rows: list[list[np.ndarray]], path: str | Path, scale: int = 4, gap: int = 2,
                  labels: list[list[str]] | None = None) -> Path:
    """Assemble une grille d'images uint8 (H, W, 3) en un PNG agrandi (plus proche voisin)."""
    Image, ImageDraw = _pil()
    h, w = rows[0][0].shape[:2]
    H, W = h * scale, w * scale
    ncol = max(len(r) for r in rows)
    lab_h = 12 if labels else 0
    sheet = Image.new("RGB", (ncol * (W + gap) + gap, len(rows) * (H + lab_h + gap) + gap), (40, 40, 48))
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            im = Image.fromarray(img).resize((W, H), Image.NEAREST)
            x, y = gap + j * (W + gap), gap + i * (H + lab_h + gap)
            sheet.paste(im, (x, y + lab_h))
            if labels:
                draw.text((x + 1, y), labels[i][j], fill=(230, 230, 230))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path
