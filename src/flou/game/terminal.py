"""Jouer au mini-jeu dans un terminal (aucune dépendance en dehors de la bibliothèque standard).

Le jeu est au tour par tour : chaque touche est un pas, et l'ennemi avance même si l'on ne fait
rien. L'image est dessinée avec des demi-blocs (deux pixels par caractère) : 64 colonnes sur
32 lignes, plus quelques lignes d'information.

- `Session`  : l'état d'une partie jouée (touches -> pas de jeu), sans aucun affichage
- `frame_to_ansi`, `rgb_to_256` : dessin de l'image avec des codes de couleur ANSI
- `decode_keys` : octets du clavier -> actions (flèches, ZQSD, WASD, espace)
"""
from __future__ import annotations

import numpy as np

from .config import ACTION_NAMES, DOWN, LEFT, NOOP, RIGHT, UP, GameConfig
from .dynamics import is_branch_state
from .env import VecMiniGame

# ---------------------------------------------------------------------------------------
# Clavier
# ---------------------------------------------------------------------------------------
_KEYS = {
    "z": UP, "w": UP, "s": DOWN, "q": LEFT, "a": LEFT, "d": RIGHT,
    " ": NOOP, ".": NOOP,
}
_ARROWS = {"A": UP, "B": DOWN, "C": RIGHT, "D": LEFT}
QUIT, RESET = "quit", "reset"
HELP = "flèches ou ZQSD / WASD : bouger · espace : ne rien faire · r : recommencer · x : quitter"


def decode_keys(data: str) -> list:
    """Transforme ce que le clavier a envoyé en une liste de jetons : une action (entier de 0 à 4),
    RESET ou QUIT. Gère les flèches (séquences ESC [ A..D), ZQSD, WASD, espace ; ignore le reste."""
    out: list = []
    i = 0
    while i < len(data):
        c = data[i]
        if c == "\x1b":
            if i + 1 < len(data) and data[i + 1] in "[O":
                # séquence d'échappement (flèches, Suppr, F1...) : on la consomme en entier
                j = i + 2
                while j < len(data) and not (data[j].isalpha() or data[j] == "~"):
                    j += 1
                if j < len(data) and j == i + 2 and data[j] in _ARROWS:
                    out.append(_ARROWS[data[j]])
                i = j + 1
                continue
            out.append(QUIT)          # Échap seul
        elif c in "\x03\x04" or c.lower() == "x":   # Ctrl-C, Ctrl-D, x
            out.append(QUIT)
        elif c.lower() == "r":
            out.append(RESET)
        elif c.lower() in _KEYS:
            out.append(_KEYS[c.lower()])
        i += 1
    return out


# ---------------------------------------------------------------------------------------
# Couleurs
# ---------------------------------------------------------------------------------------
def rgb_to_256(r: int, g: int, b: int) -> int:
    """Indice de couleur xterm 256 (cube 6x6x6) le plus proche."""
    q = lambda v: min(5, int(round(v / 51)))  # noqa: E731
    return 16 + 36 * q(r) + 6 * q(g) + q(b)


def frame_to_ansi(img: np.ndarray, truecolor: bool = True) -> str:
    """Image (H, W, 3) uint8 -> texte ANSI avec des demi-blocs « ▀ » : couleur de premier plan =
    pixel du haut, couleur de fond = pixel du bas. `truecolor=False` : palette xterm 256, pour
    les terminaux qui ne gèrent pas les couleurs 24 bits (par exemple Terminal.app de macOS)."""
    h, w, _ = img.shape
    if h % 2:
        raise ValueError("la hauteur doit être paire")
    lines = []
    for y in range(0, h, 2):
        cells = []
        prev = None
        for x in range(w):
            top, bot = tuple(int(v) for v in img[y, x]), tuple(int(v) for v in img[y + 1, x])
            if (top, bot) != prev:                     # n'émet un code que si la couleur change
                if truecolor:
                    cells.append("\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm" % (*top, *bot))
                else:
                    cells.append("\x1b[38;5;%dm\x1b[48;5;%dm" % (rgb_to_256(*top), rgb_to_256(*bot)))
                prev = (top, bot)
            cells.append("▀")
        lines.append("".join(cells) + "\x1b[0m")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Une partie jouée
# ---------------------------------------------------------------------------------------
class Session:
    """Une partie à un joueur (ou d'un bot). `press` applique une touche ; `hud` donne le texte
    d'information ; `image` l'image courante."""

    def __init__(self, cfg: GameConfig | None = None, seed: int | None = None):
        self.cfg = cfg or GameConfig()
        self.env = VecMiniGame(self.cfg, 1, seed=seed)
        self.obs = None
        self.last = "Nouvelle partie."
        self.pickups = self.hits = 0
        self.reset()

    # -----------------------------------------------------------------------------
    @property
    def t(self) -> int:
        return self.env.t

    @property
    def finished(self) -> bool:
        return self.env.t >= self.cfg.max_steps

    @property
    def score(self) -> int:
        return self.pickups - self.hits

    def reset(self, state=None) -> None:
        """Nouvelle partie ; `state` (facultatif) impose l'état initial, pour les tests."""
        obs, _ = self.env.reset(state=state)
        self.obs = obs[0]
        self.pickups = self.hits = 0
        self.last = "Nouvelle partie."

    def step(self, action: int) -> None:
        if self.finished:
            return
        obs, _, _, _, info = self.env.step(np.array([action]))
        self.obs = obs[0]
        ev, br = info["events"], int(info["branch"][0])
        notes = [f"action : {ACTION_NAMES[action]}"]
        if br >= 0:
            notes.append("l'ennemi a bifurqué à " + ("droite (A)" if br == 1 else "gauche (B)"))
        if ev.pickup[0]:
            self.pickups += 1
            notes.append("petit objet ramassé !")
        if ev.hit_enemy[0]:
            self.hits += 1
            notes.append("touché par l'ennemi")
        if ev.blocked[0]:
            notes.append("bloqué")
        self.last = " · ".join(notes)

    def press(self, token) -> bool:
        """Applique un jeton de `decode_keys`. Renvoie False pour quitter."""
        if token == QUIT:
            return False
        if token == RESET:
            self.reset()
        else:
            self.step(int(token))
        return True

    # -----------------------------------------------------------------------------
    def image(self) -> np.ndarray:
        return self.obs

    def hud(self) -> list[str]:
        cfg = self.cfg
        at_hub = bool(is_branch_state(self.env.state)[0])
        state = self.env.state
        lines = [
            f"Où naît le flou ? | p = {cfg.p:.2f} | pas {self.t}/{cfg.max_steps} | "
            f"ramassages {self.pickups} | contacts ennemi {self.hits} | score {self.score}",
            self.last,
        ]
        if self.finished:
            lines.append(f"Partie terminée : score {self.score}. r : recommencer · x : quitter")
        elif at_hub:
            lines.append(f"L'ennemi est au centre : au prochain pas il part à droite (A) avec "
                         f"probabilité {cfg.p:.2f}, à gauche (B) avec {1 - cfg.p:.2f}.")
        else:
            side = "droite" if state.eo[0] > 0 else "gauche"
            lines.append(f"L'ennemi est sur son bras de {side}, il revient vers le centre." if
                         state.ev[0] * state.eo[0] < 0 else
                         f"L'ennemi s'éloigne du centre vers la {side}.")
        lines.append(HELP)
        return lines

    def render_text(self, truecolor: bool = True) -> str:
        return frame_to_ansi(self.image(), truecolor) + "\n" + "\n".join(f"\x1b[K{l}" for l in self.hud())
