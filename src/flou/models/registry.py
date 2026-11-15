"""Construction d'un modèle à partir de sa description sérialisable (`ModelSpec`)."""
from __future__ import annotations

from .config import GPTConfig, ModelSpec, UNetConfig, VQConfig

FAMILIES = ("regression", "oracle", "diffusion", "tokens", "probe")


def default_spec(family: str, k_ctx: int = 4, K: int = 256, grid: int = 8) -> ModelSpec:
    if family == "regression":
        return ModelSpec("regression", k_ctx, unet=UNetConfig(in_ch=3 * k_ctx))
    if family == "oracle":
        return ModelSpec("oracle", k_ctx, unet=UNetConfig(in_ch=3 * k_ctx, n_extra_cond=2))
    if family == "diffusion":
        return ModelSpec("diffusion", k_ctx, unet=UNetConfig(in_ch=3 * (k_ctx + 1), sigma_cond=True))
    if family == "tokens":
        vq = VQConfig(K=K, grid=grid)
        return ModelSpec("tokens", k_ctx, vq=vq, gpt=GPTConfig(K=K, n_tokens=vq.n_tokens, ctx_frames=k_ctx))
    if family == "probe":
        return ModelSpec("probe", 1)
    raise ValueError(f"famille inconnue : {family}")


def build_model(spec: ModelSpec):
    """Importe PyTorch seulement ici : le reste du paquet reste utilisable sans lui."""
    if spec.family in ("regression", "oracle"):
        from .regression import RegressionModel
        return RegressionModel(spec.k_ctx, oracle=spec.family == "oracle", unet=spec.unet)
    if spec.family == "diffusion":
        from .diffusion import EDMDenoiser
        return EDMDenoiser(spec.k_ctx, unet=spec.unet)
    if spec.family == "tokens":
        from .tokens import TokenWorldModel
        return TokenWorldModel(spec.vq, spec.k_ctx, spec.gpt)
    if spec.family == "probe":
        from .probe_cnn import PositionProbe
        return PositionProbe()
    raise ValueError(f"famille inconnue : {spec.family}")


def expected_params(spec: ModelSpec) -> int:
    """Nombre de paramètres attendu d'après les plans (sans PyTorch)."""
    from . import params as PR
    if spec.family in ("regression", "oracle", "diffusion"):
        return PR.unet_params(spec.unet)
    if spec.family == "tokens":
        return PR.vqvae_params(spec.vq) + PR.gpt_params(spec.gpt)
    raise ValueError("pas de formule pour cette famille")
