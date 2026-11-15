"""Transformer causal (famille C) sur les jetons d'image.

Séquence d'entraînement : [jetons des k images de contexte] [jeton d'action] [jetons de l'image cible].
Le modèle prédit chaque jeton de la cible sachant ce qui précède (teacher forcing, entropie croisée) ;
les jetons de contexte sont donnés, ils ne portent pas de perte. À la génération on échantillonne
chaque jeton selon la loi prédite, à température 1 et sans top-k : toute autre valeur déformerait la loi
de bifurcation que l'on veut mesurer (cours, chap. 4).

Sans cache clé/valeur : la séquence est recalculée à chaque jeton. C'est simple et sûr ; pour 64 jetons
(321 positions) le coût reste modeste, mais le rollout de 16 pas est le poste le plus lent de la famille C.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import GPTConfig


class Block(nn.Module):
    def __init__(self, d: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, l, 3, self.n_head, d // self.n_head).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(b, l, d))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.tok_emb = nn.Embedding(cfg.K + cfg.n_actions, d)      # jetons d'image puis jetons d'action
        self.pos_emb = nn.Embedding(cfg.seq_len, d)
        self.blocks = nn.ModuleList([Block(d, cfg.n_head, cfg.dropout) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.K)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids : (B, L) long, L <= seq_len -> logits (B, L, K)."""
        b, l = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(l, device=ids.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    # ---- entraînement et génération ------------------------------------------------------------
    def sequence(self, ctx_tok: torch.Tensor, action: torch.Tensor, target_tok: torch.Tensor | None = None):
        """ctx_tok : (B, k, G) ; action : (B,) ; target_tok : (B, G) ou None -> (B, k*G + 1 [+ G])."""
        b = ctx_tok.shape[0]
        parts = [ctx_tok.reshape(b, -1), (self.cfg.K + action)[:, None]]
        if target_tok is not None:
            parts.append(target_tok)
        return torch.cat(parts, dim=1)

    def loss(self, ctx_tok, action, target_tok):
        seq = self.sequence(ctx_tok, action, target_tok)
        logits = self(seq[:, :-1])                               # (B, k*G + G, K)
        start = ctx_tok.shape[1] * ctx_tok.shape[2]               # la position de l'action prédit le 1er jeton
        pred = logits[:, start:, :]                               # (B, G, K)
        ce = F.cross_entropy(pred.float().reshape(-1, self.cfg.K), target_tok.reshape(-1))
        acc = (pred.argmax(-1) == target_tok).float().mean()
        return ce, {"ce": ce.detach(), "token_acc": acc.detach()}

    @torch.no_grad()
    def sample(self, ctx_tok, action, tau: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
        """Jetons de l'image suivante : (B, G). Température tau (1 pour mesurer la loi), sans top-k."""
        seq = self.sequence(ctx_tok, action)
        for _ in range(self.cfg.n_tokens):
            logits = self(seq)[:, -1, :].float() / tau
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
            seq = torch.cat([seq, nxt], dim=1)
        return seq[:, -self.cfg.n_tokens:]
