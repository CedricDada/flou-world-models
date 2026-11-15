#!/usr/bin/env python
"""Jouer au mini-jeu dans le terminal, ou regarder un bot jouer.

    uv run scripts/play.py                       # vous jouez : flèches ou ZQSD / WASD
    uv run scripts/play.py --p 0.5 --seed 3      # ennemi qui bifurque à pile ou face
    uv run scripts/play.py --bot expert          # regarder l'expert (Ctrl-C pour arrêter)
    uv run scripts/play.py --colors 256          # terminal sans couleurs 24 bits

Le jeu est au tour par tour : une touche = un pas, et l'ennemi avance même si vous ne faites rien.
Il faut un terminal d'au moins 64 colonnes et 40 lignes. Sur Windows, utiliser Windows Terminal ;
si les touches ne répondent pas, taper des lettres puis Entrée fonctionne aussi (« ddss »).
"""
import _bootstrap  # noqa: F401

import argparse
import os
import shutil
import sys
import time

from flou.bots import BOTS, make_bot
from flou.game import GameConfig
from flou.game.terminal import Session, decode_keys

CLEAR, HOME, HIDE, SHOW = "\x1b[2J", "\x1b[H", "\x1b[?25l", "\x1b[?25h"


def read_keys() -> str | None:
    """Lit les touches disponibles. Terminal interactif : une touche sans Entrée (Unix : termios,
    Windows : msvcrt). Entrée redirigée ou terminal exotique : une ligne à la fois. None = fin."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return None if line == "" else line
    if os.name == "nt":  # pragma: no cover - Windows
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):                       # touche spéciale : flèches
            ch = "\x1b[" + {"H": "A", "P": "B", "M": "C", "K": "D"}.get(msvcrt.getwch(), "?")
        return ch
    return os.read(sys.stdin.fileno(), 32).decode(errors="ignore") or None


class RawMode:
    """Terminal en mode « une touche à la fois » pendant le jeu (Unix) ; sans effet ailleurs."""

    def __enter__(self):
        self.old = None
        if sys.stdin.isatty() and os.name != "nt":
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=float, default=0.3, help="probabilité que l'ennemi parte à droite (A)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", type=str, default=None, help="JSON de GameConfig (facultatif)")
    ap.add_argument("--bot", choices=sorted(BOTS), default=None, help="regarder un bot au lieu de jouer")
    ap.add_argument("--delay", type=float, default=0.12, help="secondes entre deux pas d'un bot")
    ap.add_argument("--colors", choices=["auto", "truecolor", "256"], default="auto")
    args = ap.parse_args()

    base = GameConfig.from_json(args.config) if args.config else GameConfig()
    cfg = base.with_p(args.p)
    truecolor = (args.colors == "truecolor") or (
        args.colors == "auto" and os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"))
    if args.colors == "auto" and not truecolor and sys.stdout.isatty():
        print("Couleurs 24 bits non détectées : palette 256 couleurs (--colors truecolor pour forcer).")
    live = sys.stdout.isatty()                            # redessiner sur place
    cols, rows = shutil.get_terminal_size()
    if live and (cols < 64 or rows < 40):
        print(f"Terminal trop petit ({cols}x{rows}) : agrandissez-le à au moins 64x40.")
        return

    session = Session(cfg, seed=args.seed)
    bot = make_bot(args.bot, cfg) if args.bot else None
    if bot:
        import numpy as np

        brng = np.random.default_rng(0 if args.seed is None else args.seed + 1)

    def draw() -> None:
        text = session.render_text(truecolor)
        sys.stdout.write((HOME if live else "") + text + "\n")
        sys.stdout.flush()

    try:
        if live:
            sys.stdout.write(CLEAR + HIDE)
        if bot:                                           # mode spectateur
            draw()
            while not session.finished:
                time.sleep(args.delay)
                session.step(int(bot.act(session.env.state, brng)[0]))
                draw()
        else:                                             # mode joueur
            with RawMode():                               # avant le premier dessin : aucune touche perdue
                draw()
                running = True
                while running:
                    data = read_keys()
                    if data is None:
                        break
                    for token in decode_keys(data):
                        running = session.press(token)
                        if not running:
                            break
                        draw()
    except KeyboardInterrupt:
        pass
    finally:
        if live:
            sys.stdout.write(SHOW + "\n")
        print(f"Fin : {session.t} pas, {session.pickups} ramassage(s), {session.hits} contact(s) avec l'ennemi, "
              f"score {session.score}.")


if __name__ == "__main__":
    main()
