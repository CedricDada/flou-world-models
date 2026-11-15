"""Boucle d'entraînement générique : précision mixte fp16 + GradScaler, checkpoints reprenables,
journal JSONL, temps de calcul. Un entraînement est une suite de STADES (un seul pour A et B ; « vq » puis
« gpt » pour C), chacun avec son optimiseur.

Reprise : relancer la même commande ; l'entraînement repart du dernier checkpoint (`ckpt.pt`), au pas près,
avec les mêmes lots (échantillonnage sans état) et les mêmes générateurs aléatoires.

Fichiers produits dans le dossier du run : spec.json, train.json, log.jsonl, ckpt.pt, final.pt, compute.json.
NON EXÉCUTÉ dans l'environnement de rédaction (pas de PyTorch) : voir scripts/smoke_test.py.
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from . import checkpoint as ckpt


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup: int = 500
    min_lr_frac: float = 0.1
    grad_clip: float = 1.0
    amp: bool = True
    ckpt_every: int = 1000
    log_every: int = 100
    val_every: int = 2000
    ema: float = 0.0
    seed: int = 0


@dataclass
class Stage:
    name: str
    steps: int
    module: nn.Module                                   # modules dont les paramètres sont optimisés
    loss_fn: Callable[[dict], tuple]                    # lot -> (perte, journaux)
    batch_fn: Callable[[int], dict]                     # pas -> lot (sans état)
    val_fn: Callable[[], dict] | None = None            # -> {nom: valeur} sur un jeu de validation fixe
    before: Callable[[], None] | None = None            # à l'entrée du stade (aussi à la reprise)
    after_step: Callable[[int, dict], None] | None = None   # (pas, journaux) après chaque pas
    lr: float | None = None


def lr_at(step: int, total: int, base: float, warmup: int, min_frac: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))


class EMA:
    """Moyenne mobile exponentielle des paramètres (facultative)."""

    def __init__(self, module: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(module).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), module.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for s, b in zip(self.shadow.buffers(), module.buffers()):
            s.copy_(b)


def _to_float(logs: dict) -> dict:
    return {k: float(v) for k, v in logs.items() if not k.startswith("_")}


@contextlib.contextmanager
def fixed_noise(device: str, seed: int = 1234):
    """Bruit identique à chaque évaluation de validation (pertes de diffusion comparables d'un pas à l'autre)."""
    devices = [torch.cuda.current_device()] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


class Trainer:
    def __init__(self, run_dir: str | Path, top: nn.Module, stages: list[Stage], cfg: TrainConfig,
                 device: str = "cuda", meta: dict | None = None):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.top, self.stages, self.cfg, self.device = top, stages, cfg, device
        self.use_amp = bool(cfg.amp and torch.device(device).type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ema = EMA(top, cfg.ema) if cfg.ema > 0 else None
        self.meta = meta or {}
        (self.dir / "train.json").write_text(json.dumps({**asdict(cfg), **self.meta}, indent=2), encoding="utf-8")
        self.log_path = self.dir / "log.jsonl"

    # ---- journal ------------------------------------------------------------------------------
    def _log(self, **row) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    # ---- checkpoint -----------------------------------------------------------------------------
    def _save(self, stage_idx: int, step: int, opt, elapsed: float) -> None:
        ckpt.save_atomic({
            "stage": stage_idx, "step": step, "elapsed": elapsed, "model": self.top.state_dict(),
            "opt": opt.state_dict() if opt is not None else None, "scaler": self.scaler.state_dict(),
            "ema": self.ema.shadow.state_dict() if self.ema else None, "rng": ckpt.rng_state(),
        }, self.dir / "ckpt.pt")

    def _validate(self, stage: Stage) -> dict:
        if stage.val_fn is None:
            return {}
        stage.module.eval()
        with torch.no_grad(), fixed_noise(self.device):
            out = stage.val_fn()
        stage.module.train()
        return out

    # ---- boucle ---------------------------------------------------------------------------------
    def run(self, max_seconds: float | None = None, verbose: bool = True) -> dict:
        cfg, dev = self.cfg, self.device
        ckpt.seed_everything(cfg.seed)
        torch.backends.cudnn.benchmark = True
        stage_idx, step0, elapsed, resume_opt = 0, 0, 0.0, None
        path = self.dir / "ckpt.pt"
        if path.exists():
            s = ckpt.load(path, dev)
            self.top.load_state_dict(s["model"])
            stage_idx, step0, elapsed, resume_opt = s["stage"], s["step"], s["elapsed"], s["opt"]
            self.scaler.load_state_dict(s["scaler"])
            if self.ema and s["ema"] is not None:
                self.ema.shadow.load_state_dict(s["ema"])
            ckpt.set_rng_state(s["rng"])
            if verbose:
                print(f"reprise : stade {stage_idx}, pas {step0}, {elapsed / 3600:.2f} h déjà consommées")
        t_start, finished_all = time.time(), True
        if torch.device(dev).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.top.train()
        for si in range(stage_idx, len(self.stages)):
            st = self.stages[si]
            if st.before:
                st.before()
            params = [p for p in st.module.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=st.lr or cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.weight_decay)
            start = step0 if si == stage_idx else 0
            if si == stage_idx and resume_opt is not None:
                opt.load_state_dict(resume_opt)
            bad = 0
            for step in range(start, st.steps):
                if max_seconds is not None and time.time() - t_start > max_seconds:
                    self._save(si, step, opt, elapsed + time.time() - t_start)
                    finished_all = False
                    break
                lr = lr_at(step, st.steps, st.lr or cfg.lr, cfg.warmup, cfg.min_lr_frac)
                for g in opt.param_groups:
                    g["lr"] = lr
                batch = st.batch_fn(step)
                with torch.autocast(device_type=torch.device(dev).type, dtype=torch.float16, enabled=self.use_amp):
                    loss, logs = st.loss_fn(batch)
                opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                self.scaler.step(opt)
                self.scaler.update()
                if self.ema:
                    self.ema.update(self.top)
                if st.after_step:
                    st.after_step(step, logs)
                n = step + 1
                if n % cfg.log_every == 0 or n == st.steps:
                    row = {"stage": st.name, "step": n, "loss": float(loss.detach()), "lr": lr, "grad_norm": float(gnorm),
                           "elapsed_s": elapsed + time.time() - t_start, **_to_float(logs)}
                    if not math.isfinite(row["loss"]):
                        bad += 1
                        if bad >= 3:
                            raise RuntimeError(f"perte non finie à trois journaux consécutifs (pas {n}) : {row}")
                    else:
                        bad = 0
                    self._log(**row)
                    if verbose:
                        print(f"[{st.name}] {n}/{st.steps}  perte {row['loss']:.4f}  lr {lr:.2e}")
                if n % cfg.val_every == 0 or n == st.steps:
                    val = self._validate(st)
                    if val:
                        self._log(stage=st.name, step=n, **{f"val_{k}": v for k, v in val.items()})
                if n % cfg.ckpt_every == 0:
                    self._save(si, n, opt, elapsed + time.time() - t_start)
            else:
                # stade terminé normalement : le prochain démarre avec un nouvel optimiseur
                self._save(si + 1, 0, None, elapsed + time.time() - t_start)
                continue
            break                                         # interruption (max_seconds)
        total = elapsed + time.time() - t_start
        if finished_all:
            self._finish(total)
        return {"finished": finished_all, "train_seconds": total}

    def _finish(self, seconds: float) -> None:
        model = self.ema.shadow if self.ema else self.top
        torch.save({"model": model.state_dict(), "ema": bool(self.ema)}, self.dir / "final.pt")
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        info = {"gpu": gpu, "torch": torch.__version__, "train_seconds": seconds, "gpu_hours": seconds / 3600.0,
                "steps": {s.name: s.steps for s in self.stages}, "peak_memory_gb": peak,
                "parameters": sum(p.numel() for p in self.top.parameters()), "amp": self.use_amp, **self.meta}
        (self.dir / "compute.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
