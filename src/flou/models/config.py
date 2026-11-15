"""Architectures décrites par des PLANS (listes d'opérations), sans dépendance à PyTorch.

Un plan est la source unique de vérité : les modules PyTorch (`unet.py`, `vqvae.py`) sont construits en
parcourant le plan, et le compteur de paramètres (`params.py`) le parcourt aussi. La comptabilité des
connexions de saut, la source d'erreur la plus fréquente d'un U-Net écrit à la main, se teste donc en
NumPy seul (`tests/test_models_plan.py`), sans PyTorch.

Convention des opérations d'un plan de U-Net (c = canaux, r = résolution spatiale) :

    ("conv_in", cin, cout)           conv 3x3 d'entrée ; sa sortie est mise de côté (saut)
    ("res_enc", cin, cout)           bloc résiduel du chemin contractant ; sortie mise de côté
    ("down", c)                      sous-échantillonnage (conv 3x3, pas 2) ; sortie mise de côté
    ("res_mid", c, c)                bloc résiduel central
    ("res_dec", cin, cout, skip)     bloc résiduel du chemin expansif : entrée = concat(h, saut dépilé)
    ("up", c)                        sur-échantillonnage x2 (plus proche voisin) puis conv 3x3
    ("conv_out", cin, cout)          GroupNorm, SiLU, conv 3x3 de sortie (initialisée à zéro)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SIZE = 64   # images 64 x 64 (GameConfig.size) ; les modèles sont écrits pour cette taille


@dataclass(frozen=True)
class UNetConfig:
    in_ch: int = 12                       # 3 * k images de contexte (+ 3 si débruiteur : image bruitée)
    out_ch: int = 3
    channels: tuple = (48, 64, 96, 128)   # un niveau par résolution : 64, 32, 16, 8
    n_res: int = 2                        # blocs résiduels par niveau (3 dans le décodeur)
    emb_dim: int = 256                    # dimension de conditionnement (comme le débruiteur de DIAMOND)
    n_actions: int = 5
    sigma_cond: bool = False              # vrai pour le débruiteur EDM : embedding de c_noise
    n_extra_cond: int = 0                 # canaux constants de conditionnement (oracle : branche one-hot)
    groups: int = 8

    def __post_init__(self):
        if any(c % self.groups for c in self.channels):
            raise ValueError("chaque largeur doit être divisible par le nombre de groupes")
        if len(self.channels) < 2 or SIZE % (2 ** (len(self.channels) - 1)):
            raise ValueError("résolution incompatible avec le nombre de niveaux")


def unet_plan(cfg: UNetConfig) -> list[tuple]:
    """Plan du U-Net ; vérifié par `check_plan`."""
    ch = cfg.channels
    plan: list[tuple] = [("conv_in", cfg.in_ch + cfg.n_extra_cond, ch[0])]
    skips = [ch[0]]
    prev = ch[0]
    for lvl, c in enumerate(ch):
        for _ in range(cfg.n_res):
            plan.append(("res_enc", prev, c))
            skips.append(c)
            prev = c
        if lvl < len(ch) - 1:
            plan.append(("down", c))
            skips.append(c)
    plan.append(("res_mid", prev, prev))
    plan.append(("res_mid", prev, prev))
    for lvl in reversed(range(len(ch))):
        c = ch[lvl]
        for _ in range(cfg.n_res + 1):
            s = skips.pop()
            plan.append(("res_dec", prev, c, s))
            prev = c
        if lvl > 0:
            plan.append(("up", c))
    plan.append(("conv_out", prev, cfg.out_ch))
    assert not skips, "sauts non dépilés"
    return plan


def check_plan(plan: list[tuple], cfg: UNetConfig) -> None:
    """Vérifie la continuité des canaux, l'équilibre des sauts et la résolution finale (lève AssertionError)."""
    h_c = None
    r = SIZE
    stack: list[tuple[int, int]] = []     # (canaux, résolution)
    for op in plan:
        k = op[0]
        if k == "conv_in":
            assert op[1] == cfg.in_ch + cfg.n_extra_cond
            h_c = op[2]
            stack.append((h_c, r))
        elif k == "res_enc":
            assert op[1] == h_c, op
            h_c = op[2]
            stack.append((h_c, r))
        elif k == "down":
            assert op[1] == h_c
            r //= 2
            stack.append((h_c, r))
        elif k == "res_mid":
            assert op[1] == h_c == op[2]
        elif k == "res_dec":
            assert op[1] == h_c, op
            c_skip, r_skip = stack.pop()
            assert op[3] == c_skip and r_skip == r, ("saut incompatible", op, c_skip, r_skip, r)
            h_c = op[2]
        elif k == "up":
            assert op[1] == h_c
            r *= 2
        elif k == "conv_out":
            assert op[1] == h_c and op[2] == cfg.out_ch
        else:
            raise AssertionError(op)
    assert not stack and r == SIZE


# ---------------------------------------------------------------------------------------
# VQ-VAE
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VQConfig:
    K: int = 256                 # taille du codebook
    grid: int = 8                # grille de jetons grid x grid par image
    dim: int = 64                # dimension d'un code
    channels: int = 64           # largeur du codeur et du décodeur
    beta: float = 0.25           # poids du terme d'engagement (valeur de l'article VQ-VAE)
    groups: int = 8

    def __post_init__(self):
        f = SIZE // self.grid if self.grid else 0
        if self.grid < 1 or SIZE % self.grid or f < 2 or f & (f - 1):
            raise ValueError("la grille doit être 4, 8, 16 ou 32 (le quotient 64 / grille est une puissance de 2 >= 2)")

    @property
    def n_down(self) -> int:
        f = SIZE // self.grid
        return f.bit_length() - 1

    @property
    def n_tokens(self) -> int:
        return self.grid * self.grid


def vq_plan(cfg: VQConfig) -> list[tuple]:
    """Plan du VQ-VAE : codeur (conv, [bloc, down] x n_down, bloc, projection 1x1 vers `dim`),
    puis décodeur symétrique. Opérations : conv_in, block, down, proj_in, proj_out, up, conv_out."""
    c = cfg.channels
    plan: list[tuple] = [("enc_conv_in", 3, c)]
    for _ in range(cfg.n_down):
        plan += [("enc_block", c), ("enc_down", c)]
    plan += [("enc_block", c), ("enc_proj", c, cfg.dim)]
    plan += [("dec_proj", cfg.dim, c), ("dec_block", c)]
    for _ in range(cfg.n_down):
        plan += [("dec_up", c), ("dec_block", c)]
    plan += [("dec_conv_out", c, 3)]
    return plan


@dataclass(frozen=True)
class GPTConfig:
    K: int = 256                 # vocabulaire des jetons d'image (sortie)
    n_actions: int = 5
    n_tokens: int = 64           # G : jetons par image
    ctx_frames: int = 4
    d: int = 320
    n_layer: int = 4
    n_head: int = 5
    dropout: float = 0.0

    @property
    def seq_len(self) -> int:
        """Longueur de la séquence d'entraînement : contexte, jeton d'action, image cible (le dernier
        jeton de la cible n'est pas une entrée : on le prédit)."""
        return self.ctx_frames * self.n_tokens + 1 + self.n_tokens

    def __post_init__(self):
        if self.d % self.n_head:
            raise ValueError("d doit être divisible par n_head")


@dataclass(frozen=True)
class ModelSpec:
    """Choix d'architecture d'une famille (sérialisé dans le dossier du run)."""
    family: str                        # regression | oracle | diffusion | tokens | probe
    k_ctx: int = 4
    unet: UNetConfig | None = None
    vq: VQConfig | None = None
    gpt: GPTConfig | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
