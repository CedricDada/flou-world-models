"""Modèles : A régression, B diffusion EDM, C jetons discrets (VQ-VAE + Transformer causal).

Importer ce paquet n'importe PAS PyTorch : `config`, `params`, `edm_math` et `registry.default_spec`
fonctionnent en NumPy seul. Les modules PyTorch (`unet`, `regression`, `diffusion`, `vqvae`, `gpt`,
`tokens`, `probe_cnn`) ne sont importés que par `registry.build_model`.
"""
