"""Construit, pour un `RunSpec`, le modèle et la suite de stades d'entraînement.

* regression, oracle, diffusion : un stade « main » (perte du modèle) ;
* tokens : stade « vq » (VQ-VAE sur les images cibles, avec réinitialisation périodique des codes morts),
  puis stade « gpt » (le VQ-VAE est gelé, toutes les images de la banque sont converties en jetons, le
  Transformer apprend la loi du jeton suivant) ;
* probe : sonde de position (CNN) entraînée sur des images dont l'état exact est connu.

La validation utilise un jeu fixe d'indices tirés une fois (`val_n` exemples) : courbes comparables.
"""
from __future__ import annotations

import torch

from ..models.registry import build_model
from .frames import FrameBank
from .runs import RunSpec
from .trainer import Stage, TrainConfig


def _val_indices(bank: FrameBank, n: int, frames: bool = False) -> torch.Tensor:
    g = torch.Generator(device=bank.device)
    g.manual_seed(20_240_607)
    hi = bank.n_frames if frames else bank.n
    return torch.randint(0, hi, (min(n, hi),), generator=g, device=bank.device)


def _chunks(idx: torch.Tensor, size: int):
    for i in range(0, len(idx), size):
        yield idx[i:i + size]


def build_task(spec: RunSpec, bank: FrameBank, val_bank: FrameBank | None, cfg: TrainConfig,
               device: str = "cuda", val_n: int = 2048, restart_every: int = 200):
    """Renvoie (modèle complet, liste de stades)."""
    model = build_model(spec.model_spec()).to(device)
    bs, seed = cfg.batch_size, cfg.seed
    batch_fn = lambda step: bank.batch(bank.sample_idx(step, bs, seed))
    val_idx = _val_indices(val_bank, val_n) if val_bank is not None else None

    if spec.family in ("regression", "oracle", "diffusion"):
        def loss_fn(b):
            return model.loss(b["ctx"], b["action"], b["target"], b["branch"])

        def val_fn():
            tot, n = 0.0, 0
            for ids in _chunks(val_idx, bs):
                b = val_bank.batch(ids)
                loss, _ = model.loss(b["ctx"], b["action"], b["target"], b["branch"])
                tot, n = tot + float(loss) * len(ids), n + len(ids)
            return {"loss": tot / n}

        return model, [Stage("main", spec.steps, model, loss_fn, batch_fn, val_fn if val_bank else None)]

    if spec.family == "tokens":
        steps = spec.stage_steps()

        def vq_loss(b):
            loss, logs = model.vq.loss(b["target"])
            return loss, logs

        def vq_val():
            tot, n = 0.0, 0
            for ids in _chunks(val_idx, bs):
                _, _, recon, _, _ = model.vq(val_bank.batch(ids)["target"])
                tot, n = tot + float(recon) * len(ids), n + len(ids)
            return {"recon": tot / n}

        def vq_after(step, logs):
            if restart_every and (step + 1) % restart_every == 0 and step + 1 < 0.8 * steps["vq"]:
                n = model.vq.restart_dead_codes(logs["_z"])
                if n:
                    print(f"[vq] pas {step + 1} : {n} codes réinitialisés")

        state = {}

        def gpt_before():
            for p in model.vq.parameters():
                p.requires_grad_(False)
            model.vq.eval()
            state["train"] = model.tokenize_images(bank.frames)
            state["val"] = model.tokenize_images(val_bank.frames) if val_bank is not None else None

        def gpt_loss(b):
            ctx_tok, tgt_tok = bank.token_batch(b["idx"], state["train"])
            return model.gpt.loss(ctx_tok, b["action"], tgt_tok)

        def gpt_val():
            tot, n = 0.0, 0
            for ids in _chunks(val_idx, bs):
                b = val_bank.batch(ids)
                ctx_tok, tgt_tok = val_bank.token_batch(ids, state["val"])
                loss, _ = model.gpt.loss(ctx_tok, b["action"], tgt_tok)
                tot, n = tot + float(loss) * len(ids), n + len(ids)
            return {"ce": tot / n}

        return model, [
            Stage("vq", steps["vq"], model.vq, vq_loss, batch_fn, vq_val if val_bank else None, after_step=vq_after),
            Stage("gpt", steps["gpt"], model.gpt, gpt_loss, batch_fn, gpt_val if val_bank else None, before=gpt_before),
        ]
    raise ValueError(spec.family)


def build_probe_task(steps: int, bank: FrameBank, val_bank: FrameBank, cfg: TrainConfig, device: str = "cuda",
                     val_n: int = 4096):
    """Sonde de position : (modèle, stades). `bank` et `val_bank` doivent avoir `with_positions=True`."""
    from ..models.registry import default_spec
    model = build_model(default_spec("probe")).to(device)
    bs, seed = cfg.batch_size, cfg.seed
    batch_fn = lambda step: bank.frame_batch(bank.sample_frames(step, bs, seed))
    val_idx = _val_indices(val_bank, val_n, frames=True)

    def loss_fn(b):
        return model.loss(b["x"], b["pos"])

    def val_fn():
        se, n = 0.0, 0
        for ids in _chunks(val_idx, 512):
            b = val_bank.frame_batch(ids)
            pred = model(b["x"])
            se += float(((pred - b["pos"]) ** 2).sum(-1).sum())
            n += pred.shape[0] * pred.shape[1]
        return {"rmse_px": (se / n) ** 0.5}

    return model, [Stage("main", steps, model, loss_fn, batch_fn, val_fn)]
