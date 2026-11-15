#!/usr/bin/env python
"""Test de fumée du code PyTorch : à lancer UNE FOIS sur la machine d'entraînement (Colab T4, poste GPU) avant tout run.

Le code PyTorch (modèles, boucle d'entraînement, évaluation) n'a PAS pu être exécuté là où il a été écrit (pas de
PyTorch). Ce script le fait travailler de bout en bout sur de minuscules données, en quelques minutes, et dit
précisément ce qui casse. Ce qu'il vérifie :

  1. le nombre de paramètres de chaque modèle PyTorch égale celui que prédit le plan (budget ~ 6 M pour A, B, C) ;
  2. chaque famille s'entraîne quelques pas (perte finie) avec journal, validation et checkpoint ;
  3. la REPRISE depuis un checkpoint fonctionne (interruption simulée en plein entraînement) ;
  4. l'évaluation complète tourne sur chaque famille (échantillonnage, rollouts, planches, tableaux) ;
  5. l'extension « décision » tourne sur chaque famille : niveau 1 (decision.json) et planificateur (planning.json) ;
  6. avec --time : temps par pas à la taille réelle -> reports/timing.json (lu par scripts/plan.py), et débit
     d'inférence (images par seconde) -> durée estimée du niveau 2 de l'extension « décision ».

Exemples
    uv run scripts/smoke_test.py                       # vérifications 1 à 4, quelques minutes sur GPU
    uv run scripts/smoke_test.py --time                # + mesure des temps par pas (lot 64)
    uv run scripts/smoke_test.py --device cpu          # sans GPU : lent, mais vérifie la plomberie
    uv run scripts/smoke_test.py --only regression diffusion
Code de sortie 1 si une vérification échoue.
"""
import _bootstrap

import argparse
import json
import shutil
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path

FAMILIES = ("regression", "oracle", "diffusion", "tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    ap.add_argument("--time", action="store_true", help="mesure le temps par pas à la taille réelle")
    ap.add_argument("--batch-size", type=int, default=64, help="lot pour --time")
    ap.add_argument("--time-steps", type=int, default=15)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--keep", action="store_true", help="garde le dossier temporaire")
    args = ap.parse_args()

    torch = _bootstrap.require_torch()                      # message clair si PyTorch manque
    from flou.data import build_bifurcation_set, generate_split
    from flou.eval import instrument as _unused  # noqa: F401  (vérifie l'import du paquet d'évaluation)
    from flou.game import GameConfig
    from flou.models.registry import build_model, default_spec, expected_params
    from flou.train import runs as R
    from flou.train.evaluation import evaluate_run
    from flou.train.frames import FrameBank
    from flou.train.tasks import build_task
    from flou.train.trainer import Trainer, TrainConfig

    dev = args.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA indisponible : passer --device cpu ou changer d'environnement (Colab : GPU T4)")
    gpu = torch.cuda.get_device_name(0) if dev.startswith("cuda") else "cpu"
    print(f"PyTorch {torch.__version__} sur {gpu}")

    fails: list[str] = []

    def check(name, fn):
        t = time.time()
        try:
            info = fn()
            print(f"[ OK ] {name}" + (f" — {info}" if info else "") + f"  ({time.time() - t:.1f} s)", flush=True)
            return True
        except Exception:                                                  # noqa: BLE001 (on veut tout rapporter)
            fails.append(name)
            print(f"[FAIL] {name}", flush=True)
            traceback.print_exc(limit=6)
            return False

    # ---- 1. nombre de paramètres ----------------------------------------------------------------------------
    def params(family):
        spec = default_spec(family)
        got = sum(p.numel() for p in build_model(spec).parameters())
        exp = expected_params(spec)
        if got != exp:
            raise AssertionError(f"{family} : {got} paramètres PyTorch contre {exp} attendus par le plan")
        return f"{got:,} paramètres (= plan)".replace(",", " ")

    for fam in args.only:
        check(f"paramètres {fam}", lambda fam=fam: params(fam))
    if len(args.only) == len(FAMILIES):
        sizes = {f: expected_params(default_spec(f)) for f in FAMILIES}
        spread = max(sizes.values()) / min(sizes.values()) - 1
        print(f"       écart de taille entre familles : {100 * spread:.1f} % (budget apparié visé : quelques %)")

    def probe_shape():
        m = build_model(default_spec("probe")).to(dev)
        out = m(torch.zeros(2, 3, 64, 64, device=dev))
        assert tuple(out.shape) == (2, 4, 2), out.shape
        return f"{sum(p.numel() for p in m.parameters()):,} paramètres".replace(",", " ")

    check("sonde CNN (forme de sortie)", probe_shape)

    # ---- données minuscules -----------------------------------------------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="flou_smoke_"))
    data, runs = tmp / "data", tmp / "runs"
    cfg = GameConfig().with_p(0.3)
    generate_split(cfg, "train", ["random", "greedy"], 2000, 0, data, n_envs=32)
    generate_split(cfg, "val", ["random", "greedy"], 1000, 0, data, n_envs=32)
    bs = build_bifurcation_set(cfg, n=60, seed=0)
    bs.save(R.data_dir_for(data, 0.3) / "test_bifurcation.npz", {"config": cfg.to_dict()})
    build_bifurcation_set(cfg, n=200, seed=1000).save(R.data_dir_for(data, 0.3) / "test_decision.npz",
                                                      {"config": cfg.to_dict()})
    bank = FrameBank(R.train_files(data, 0.3, "train"), 4, dev)
    val_bank = FrameBank(R.train_files(data, 0.3, "val"), 4, dev)
    print(f"données de fumée : {bank.n} transitions d'entraînement, {val_bank.n} de validation, {len(bs)} états de test")

    # ---- 2-4. entraînement, reprise, évaluation ----------------------------------------------------------------
    def tiny_spec(family):
        return R.RunSpec(family, "core", 0.3, 0, K=256 if family == "tokens" else None, steps=6, batch_size=8)

    def train_cfg(spec):
        return TrainConfig(batch_size=8, lr=spec.lr, seed=0, amp=dev.startswith("cuda"), ckpt_every=2, val_every=3,
                           log_every=1, warmup=2)

    def train(family):
        spec = tiny_spec(family)
        d = runs / spec.run_id
        spec.save(d)
        c = train_cfg(spec)
        model, stages = build_task(spec, bank, val_bank, c, dev, val_n=32)
        out = Trainer(d, model, stages, c, dev, meta={"run_id": spec.run_id}).run(verbose=False)
        assert out["finished"] and (d / "final.pt").exists() and (d / "compute.json").exists()
        rows = [json.loads(line) for line in open(d / "log.jsonl", encoding="utf-8")]
        losses = [r["loss"] for r in rows if "loss" in r]
        assert losses and all(x == x and abs(x) < 1e6 for x in losses), f"pertes non finies : {losses}"
        assert any("val_loss" in r or "val_recon" in r or "val_ce" in r for r in rows), "aucune validation journalisée"
        return f"{len(losses)} journaux, perte {losses[0]:.3f} -> {losses[-1]:.3f}"

    def resume():
        spec = replace(tiny_spec("regression"), group="control", p=0.3, note="smoke-resume")
        d = runs / "resume-test"
        c = train_cfg(spec)
        model, stages = build_task(spec, bank, val_bank, c, dev, val_n=32)

        def boom(step, logs):
            if step == 3:                                                  # après le checkpoint du pas 2
                raise RuntimeError("interruption simulée")
        stages[0].after_step = boom
        try:
            Trainer(d, model, stages, c, dev).run(verbose=False)
            raise AssertionError("l'interruption simulée n'a pas eu lieu")
        except RuntimeError as e:
            assert "simulée" in str(e)
        assert (d / "ckpt.pt").exists() and not (d / "final.pt").exists()
        model2, stages2 = build_task(spec, bank, val_bank, c, dev, val_n=32)
        out = Trainer(d, model2, stages2, c, dev).run(verbose=False)
        assert out["finished"] and (d / "final.pt").exists()
        steps = [json.loads(line)["step"] for line in open(d / "log.jsonl", encoding="utf-8") if '"loss"' in line]
        assert steps[-1] == 6 and steps.count(3) == 2 and steps.count(4) == 1, f"reprise incohérente : {steps}"
        return "interruption au pas 3, reprise depuis le checkpoint du pas 2 (le pas 3 est rejoué)"

    def evaluate(family):
        spec = tiny_spec(family)
        res = evaluate_run(runs / spec.run_id, data, 0.17, device=dev, n_list=(1, 3), rollout_n=8, horizon=4,
                           verbose=False, amp=dev.startswith("cuda"))
        assert (runs / spec.run_id / "eval.json").exists() and (runs / spec.run_id / "eval_arrays.npz").exists()
        v = res["variants"][-1]["summary"]
        return f"{len(res['variants'])} variante(s), n = {v['n']} images, flou {v['blur']['value']:.2f}"

    def decision(family):
        """Extension « décision » (docs/EXTENSION_DECISION.md) : niveau 1 puis planificateur, en tout petit."""
        from flou.eval.planning import PlanConfig
        from flou.train.evaluation import evaluate_decision_run, plan_run
        d = runs / tiny_spec(family).run_id
        amp = dev.startswith("cuda")
        res = evaluate_decision_run(d, data, device=dev, ks=(1, 2), m=2, n_list=(1, 3), amp=amp, verbose=False)
        assert (d / "decision.json").exists() and res["variants"], "decision.json vide"
        r1 = res["variants"][-1]["risk"]["per_k"]["1"]
        pc = PlanConfig(candidates=4, horizon=2, samples=2)
        out = plan_run(d, data, device=dev, pc=pc, n_games=2, game_steps=2, amp=amp, verbose=False)
        pl = out["variants"][0]["play"]
        want = 2 * 2 * 4 * 2 * (1 if family == "regression" else 2)
        assert (d / "planning.json").exists() and pl["predictions"] == want, (pl["predictions"], want)
        return (f"niveau 1 : {r1['n_ambiguous']} états ambigus, erreur {r1['soft']['amb']['mean']:.2f} ; "
                f"planificateur : {pl['predictions']} images imaginées, agent perdu {100 * pl['agent_lost']:.0f} %")

    trained = []
    for fam in args.only:
        if check(f"entraînement (6 pas) {fam}", lambda fam=fam: train(fam)):
            trained.append(fam)
    if "regression" in args.only:
        check("reprise depuis un checkpoint", resume)
    for fam in trained:
        check(f"évaluation complète {fam}", lambda fam=fam: evaluate(fam))
    for fam in trained:
        if fam != "oracle":                                              # l'oracle reçoit la branche : hors sujet
            check(f"extension décision {fam}", lambda fam=fam: decision(fam))

    # ---- 5. temps par pas ------------------------------------------------------------------------------------------
    if args.time:
        sec, mem = {}, {}

        def time_family(family):
            spec = R.RunSpec(family, "core", 0.3, 0, K=256 if family == "tokens" else None)
            c = TrainConfig(batch_size=args.batch_size, lr=spec.lr, seed=0, amp=dev.startswith("cuda"))
            model, stages = build_task(spec, bank, val_bank, c, dev, val_n=32)
            for st in stages:
                if st.before:
                    st.before()
                params = [p for p in st.module.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=1e-4)
                scaler = torch.amp.GradScaler("cuda", enabled=c.amp)
                if dev.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                for i in range(3 + args.time_steps):
                    if i == 3:
                        if dev.startswith("cuda"):
                            torch.cuda.synchronize()
                        t0 = time.time()
                    b = st.batch_fn(i)
                    with torch.autocast(device_type=torch.device(dev).type, dtype=torch.float16, enabled=c.amp):
                        loss, _ = st.loss_fn(b)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                if dev.startswith("cuda"):
                    torch.cuda.synchronize()
                key = family if family != "tokens" else f"tokens-{st.name}"
                sec[key] = (time.time() - t0) / args.time_steps
                mem[key] = torch.cuda.max_memory_allocated() / 1e9 if dev.startswith("cuda") else 0.0
            return ", ".join(f"{k} {v:.3f} s/pas" for k, v in sec.items() if k.startswith(family))

        for fam in args.only:
            check(f"temps par pas {fam} (lot {args.batch_size})", lambda fam=fam: time_family(fam))

        # débit d'inférence : sert à estimer le niveau 2 de l'extension « décision » (planification)
        ips = {}

        def time_inference(family, n_img=512, reps=3):
            import numpy as np
            from flou.eval.predictors import Batch
            from flou.train.predictors import TorchPredictor
            spec = R.RunSpec(family, "core", 0.3, 0, K=256 if family == "tokens" else None)
            model = build_model(spec.model_spec()).to(dev)
            kw = {"n_steps": 3} if family == "diffusion" else {}
            pred = TorchPredictor(model, dev, amp=dev.startswith("cuda"), **kw)
            rng = np.random.default_rng(0)
            ctx = rng.integers(0, 255, (n_img, 4, 64, 64, 3), dtype=np.uint8)
            b = Batch(np.arange(n_img), ctx, rng.integers(0, 5, n_img), 0)
            pred.predict(b, rng)                                         # échauffement
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(reps):
                pred.predict(b, rng)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            ips[family] = reps * n_img / (time.time() - t0)
            return f"{ips[family]:.0f} images/s" + (" (n = 3 pas d'Euler)" if family == "diffusion" else "")

        for fam in args.only:
            if fam != "oracle":
                check(f"débit d'inférence {fam}", lambda fam=fam: time_inference(fam))
        if ips:
            from flou.eval.planning import PlanConfig
            pc, games = PlanConfig(), 50
            print(f"\nextension « décision », niveau 2, réglages par défaut ({games} parties de 100 pas, J = {pc.candidates}, "
                  f"H = {pc.horizon}, m = {pc.samples}) : durée estimée PAR RUN (inférence seule, sans la lecture des images)")
            for fam, v in ips.items():
                n_img = games * 100 * pc.candidates * pc.horizon * (1 if fam == "regression" else pc.samples)
                print(f"  {fam:<11} {n_img / 1e6:5.1f} millions d'images -> {n_img / v / 3600:5.2f} h")
            print("  (réduire avec --games / --candidates de scripts/evaluate_decision.py, identiques pour tous les runs)")
        if len(sec) == 5:                                                # toutes les familles : on peut estimer le plan
            args.out.mkdir(parents=True, exist_ok=True)
            rep = {"device": dev, "gpu": gpu, "torch": torch.__version__, "batch_size": args.batch_size,
                   "amp": dev.startswith("cuda"), "time_steps": args.time_steps, "seconds_per_step": sec,
                   "peak_memory_gb": mem, "inference_images_per_second": ips,
                   "note": "mesuré sur de petites données et sans le coût des validations/checkpoints : "
                           "prévoir une marge (~10 %)"}
            (args.out / "timing.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
            plan = R.build_plan(batch_size=args.batch_size)
            print(f"\nplan par défaut (20 000 pas par run) : {R.estimate_hours(plan, sec):.1f} h de GPU estimées "
                  f"(sans marge) ; détails : uv run scripts/plan.py --timing {args.out / 'timing.json'}")
        elif args.time:
            print("temps incomplets (toutes les familles sont nécessaires pour estimer le plan) : timing.json non écrit")

    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"dossier conservé : {tmp}")
    print()
    if fails:
        print(f"{len(fails)} vérification(s) en échec : " + "; ".join(fails))
        raise SystemExit(1)
    print("toutes les vérifications passent : le code PyTorch tourne de bout en bout sur ces petites données")
    print("(ce que ce test ne dit PAS : que les modèles apprennent bien, ni combien de temps prend un vrai run)")


if __name__ == "__main__":
    main()
