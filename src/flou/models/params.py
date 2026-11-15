"""Comptage de paramètres et de multiplications-additions, calculé sur les plans de `config.py`.

Les formules reproduisent exactement les couches construites dans `unet.py`, `vqvae.py` et `gpt.py` ;
`scripts/smoke_test.py` compare ce compte au nombre réel de paramètres PyTorch (écart attendu : 0).
"""
from __future__ import annotations

from .config import GPTConfig, SIZE, UNetConfig, VQConfig, unet_plan, vq_plan


def conv(cin: int, cout: int, k: int) -> int:
    return cin * cout * k * k + cout


def linear(i: int, o: int) -> int:
    return i * o + o


def groupnorm(c: int, affine: bool = True) -> int:
    return 2 * c if affine else 0


def res_block(cin: int, cout: int, emb: int) -> int:
    """norm1 (affine) + conv1 + norm2 (sans paramètres, AdaGN) + projection (scale, shift) + conv2 (+ saut 1x1)."""
    n = groupnorm(cin) + conv(cin, cout, 3) + linear(emb, 2 * cout) + conv(cout, cout, 3)
    if cin != cout:
        n += conv(cin, cout, 1)
    return n


def unet_params(cfg: UNetConfig) -> int:
    n = 0
    for op in unet_plan(cfg):
        k = op[0]
        if k == "conv_in":
            n += conv(op[1], op[2], 3)
        elif k in ("res_enc", "res_mid"):
            n += res_block(op[1], op[2], cfg.emb_dim)
        elif k == "res_dec":
            n += res_block(op[1] + op[3], op[2], cfg.emb_dim)
        elif k in ("down", "up"):
            n += conv(op[1], op[1], 3)
        elif k == "conv_out":
            n += groupnorm(op[1]) + conv(op[1], op[2], 3)
    n += cfg.n_actions * cfg.emb_dim                                   # embedding de l'action
    if cfg.sigma_cond:
        n += linear(cfg.emb_dim, cfg.emb_dim) + linear(cfg.emb_dim, cfg.emb_dim)   # MLP du bruit
    return n


def unet_macs(cfg: UNetConfig) -> int:
    """Multiplications-additions d'une passe avant, un exemple (convolutions seulement)."""
    r = SIZE
    macs = 0
    stack_r = []
    for op in unet_plan(cfg):
        k = op[0]
        if k == "conv_in":
            macs += op[1] * op[2] * 9 * r * r
            stack_r.append(r)
        elif k in ("res_enc", "res_mid"):
            macs += (op[1] * op[2] * 9 + op[2] * op[2] * 9 + (op[1] * op[2] if op[1] != op[2] else 0)) * r * r
            if k == "res_enc":
                stack_r.append(r)
        elif k == "res_dec":
            cin = op[1] + op[3]
            macs += (cin * op[2] * 9 + op[2] * op[2] * 9 + cin * op[2]) * r * r
            stack_r.pop()
        elif k == "down":
            r //= 2
            macs += op[1] * op[1] * 9 * r * r
            stack_r.append(r)
        elif k == "up":
            r *= 2
            macs += op[1] * op[1] * 9 * r * r
        elif k == "conv_out":
            macs += op[1] * op[2] * 9 * r * r
    return macs


def vq_block(c: int) -> int:
    return 2 * groupnorm(c) + 2 * conv(c, c, 3)


def vqvae_params(cfg: VQConfig) -> int:
    n = cfg.K * cfg.dim
    for op in vq_plan(cfg):
        k = op[0]
        if k in ("enc_conv_in", "dec_conv_out"):
            n += conv(op[1], op[2], 3) + (groupnorm(op[1]) if k == "dec_conv_out" else 0)
        elif k in ("enc_block", "dec_block"):
            n += vq_block(op[1])
        elif k in ("enc_down", "dec_up"):
            n += conv(op[1], op[1], 3)
        elif k in ("enc_proj", "dec_proj"):
            n += conv(op[1], op[2], 1)
    return n


def gpt_params(cfg: GPTConfig) -> int:
    d = cfg.d
    block = 2 * (2 * d) + linear(d, 3 * d) + linear(d, d) + linear(d, 4 * d) + linear(4 * d, d)
    n = (cfg.K + cfg.n_actions) * d                  # embedding des jetons d'image et d'action
    n += cfg.seq_len * d                             # positions apprises
    n += cfg.n_layer * block
    n += 2 * d + linear(d, cfg.K)                    # LayerNorm finale + tête
    return n


def human(n: int) -> str:
    return f"{n / 1e6:.2f} M"
