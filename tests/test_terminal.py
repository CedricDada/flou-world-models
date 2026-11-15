import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import make_state

from flou.game import DOWN, LEFT, NOOP, RIGHT, UP, render
from flou.game.terminal import QUIT, RESET, Session, decode_keys, frame_to_ansi, rgb_to_256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "play.py"


# ---------------------------------------------------------------------------------------
# Clavier
# ---------------------------------------------------------------------------------------
def test_decode_arrows_zqsd_wasd_space():
    assert decode_keys("\x1b[A\x1b[B\x1b[C\x1b[D") == [UP, DOWN, RIGHT, LEFT]
    assert decode_keys("\x1bOA\x1bOD") == [UP, LEFT]               # variante « application »
    assert decode_keys("zqsd") == [UP, LEFT, DOWN, RIGHT]
    assert decode_keys("wasd") == [UP, LEFT, DOWN, RIGHT]
    assert decode_keys("ZQSD") == [UP, LEFT, DOWN, RIGHT]
    assert decode_keys(" .") == [NOOP, NOOP]
    assert decode_keys("r") == [RESET]


def test_decode_quit_and_unknown_keys():
    assert decode_keys("x") == [QUIT] and decode_keys("\x03") == [QUIT] and decode_keys("\x1b") == [QUIT]
    assert decode_keys("hello") == []                               # touches inconnues ignorées
    assert decode_keys("\x1b[3~d") == [RIGHT]                       # Suppr : consommée, ne quitte pas
    assert decode_keys("\x1b[1;5C") == []                           # Ctrl+flèche : ignorée


# ---------------------------------------------------------------------------------------
# Dessin
# ---------------------------------------------------------------------------------------
def parse_ansi(text: str) -> np.ndarray:
    """Relit un dessin truecolor et reconstruit l'image (H, W, 3) : vérifie l'aller-retour."""
    rows = []
    for line in text.split("\n"):
        fg = bg = None
        top, bottom = [], []
        for m in re.finditer(r"\x1b\[38;2;(\d+);(\d+);(\d+)m|\x1b\[48;2;(\d+);(\d+);(\d+)m|▀", line):
            if m.group(1):
                fg = tuple(int(m.group(i)) for i in (1, 2, 3))
            elif m.group(4):
                bg = tuple(int(m.group(i)) for i in (4, 5, 6))
            else:
                top.append(fg), bottom.append(bg)
        rows += [top, bottom]
    return np.array(rows, dtype=np.uint8)


def test_frame_to_ansi_roundtrip(cfg):
    img = render(make_state(cfg), cfg)[0]
    text = frame_to_ansi(img)
    assert text.count("\n") + 1 == 32
    assert np.array_equal(parse_ansi(text), img)


def test_frame_to_ansi_256_colors(cfg):
    img = render(make_state(cfg), cfg)[0]
    text = frame_to_ansi(img, truecolor=False)
    assert "38;2;" not in text and "38;5;" in text
    assert text.count("▀") == 64 * 32
    assert (rgb_to_256(0, 0, 0), rgb_to_256(255, 0, 0), rgb_to_256(255, 255, 255)) == (16, 196, 231)
    assert rgb_to_256(*cfg.color_agent) != rgb_to_256(*cfg.color_enemy) != rgb_to_256(*cfg.color_gem)
    assert len({rgb_to_256(*c) for c in (cfg.color_bg, cfg.color_wall, cfg.color_obstacle, cfg.color_enemy,
                                          cfg.color_agent, cfg.color_gem)}) == 6   # six teintes distinctes


# ---------------------------------------------------------------------------------------
# Une partie jouée
# ---------------------------------------------------------------------------------------
def test_session_moves_scores_and_ends(cfg):
    s = Session(cfg, seed=0)
    s.reset(state=make_state(cfg, ax=30, ay=10, eo=2, ev=1, gx=6, gy=6))
    assert s.press(RIGHT) and s.press(DOWN)
    state = s.env.state
    assert (state.ax[0], state.ay[0]) == (32, 12) and s.t == 2
    assert "action : bas" in s.last
    assert np.array_equal(s.image(), render(state, cfg)[0])
    # ramassage : l'agent chevauche déjà le petit objet
    s.reset(state=make_state(cfg, ax=4, ay=4, gx=6, gy=6))
    s.press(NOOP)
    assert s.pickups == 1 and s.score == 1 and "ramassé" in s.last
    # fin de partie : les pas suivants sont ignorés
    for _ in range(cfg.max_steps):
        s.press(NOOP)
    assert s.finished and s.t == cfg.max_steps
    assert any("terminée" in line for line in s.hud())
    assert s.press(NOOP) and s.t == cfg.max_steps
    assert s.press(RESET) and s.t == 0 and s.score == 0
    assert s.press(QUIT) is False


def test_hud_announces_the_bifurcation(cfg):
    s = Session(cfg, seed=0)
    s.reset(state=make_state(cfg, eo=0, ev=-1))
    assert "au centre" in s.hud()[2] and "0.30" in s.hud()[2] and "0.70" in s.hud()[2]
    s.press(NOOP)
    assert "bifurqué" in s.last and "au centre" not in s.hud()[2]
    s.reset(state=make_state(cfg, eo=2, ev=-1))
    assert "revient" in s.hud()[2]
    s.reset(state=make_state(cfg, eo=2, ev=1))
    assert "s'éloigne" in s.hud()[2]


# ---------------------------------------------------------------------------------------
# Le script, en conditions réelles
# ---------------------------------------------------------------------------------------
def run_script(*args, stdin="", timeout=30):
    return subprocess.run([sys.executable, str(SCRIPT), *args], input=stdin, capture_output=True,
                          text=True, timeout=timeout)


def test_script_with_piped_keys():
    r = run_script("--seed", "1", "--colors", "256", stdin="dd\nss\nx\n")
    assert r.returncode == 0, r.stderr
    assert "pas 4/100" in r.stdout and "Fin : 4 pas" in r.stdout


def test_script_bot_mode_plays_a_full_episode():
    r = run_script("--bot", "expert", "--seed", "1", "--delay", "0", "--colors", "256")
    assert r.returncode == 0, r.stderr
    assert "Partie terminée" in r.stdout and "Fin : 100 pas" in r.stdout


@pytest.mark.skipif(os.name == "nt", reason="terminal pseudo-interactif : Unix seulement")
def test_script_in_a_real_terminal_reads_single_keypresses():
    import fcntl
    import pty
    import select
    import struct
    import termios

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 100, 0, 0))   # 100 colonnes x 50 lignes
    proc = subprocess.Popen([sys.executable, str(SCRIPT), "--seed", "1", "--colors", "256"],
                            stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    seen = ""

    def wait_for(text, timeout=10.0):
        nonlocal seen
        end = time.time() + timeout
        while text not in seen and time.time() < end:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    seen += os.read(master, 65536).decode(errors="ignore")
                except OSError:
                    break
        return text in seen

    try:
        assert wait_for("pas 0/100"), seen[-300:]
        os.write(master, b"d")                    # une touche, sans Entrée
        assert wait_for("pas 1/100"), seen[-300:]
        os.write(master, b"\x1b[B")               # flèche vers le bas
        assert wait_for("pas 2/100") and "action : bas" in seen
        os.write(master, b"x")
        assert proc.wait(timeout=10) == 0
        assert wait_for("Fin : 2 pas")
    finally:
        if proc.poll() is None:
            proc.kill()
        os.close(master)
