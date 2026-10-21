"""Loop de treino compartilhado pelos notebooks e pela linha de comando.

Nos notebooks:
    run = train(model, ds, name="02-atencao", steps=2000, lr=1e-3)
    run.history  # lista de dicts com step, train_loss, val_loss, elapsed_s

Na linha de comando (para treinos longos, como os dos notebooks 07 e 08):
    python -m lab.train --name shakespeare-10m --dataset shakespeare --n_layer 6 --n_head 6 --d_model 384 \\
        --block_size 256 --dropout 0.2 --attn batched --steps 5000 --lr 3e-4 --batch_size 64 --save

Cada execução grava em runs/<nome>/: config.json, metrics.csv, sample.txt e, com save=True, model.pt.
Defina LAB_MAX_STEPS=N no ambiente para limitar todos os treinos a N passos (teste rápido).
"""

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import torch

from lab.data import CharTokenizer, Dataset, get_batch, shakespeare, tinystories
from lab.model import GPT, GPTConfig, count_params

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = 1337):
    torch.manual_seed(seed)


@dataclass
class TrainConfig:
    name: str = "base"
    steps: int = 2000
    batch_size: int = 32
    block_size: int | None = None  # None = o block_size do modelo
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 0.0          # 0 = sem clipping
    eval_every: int = 100
    eval_batches: int = 20
    seed: int = 1337
    device: str | None = None       # None = MPS se houver
    sample_tokens: int = 300
    sample_prompt: str = "\n"
    save: bool = False              # grava model.pt com pesos, configuração e tokenizer


@dataclass
class Run:
    name: str
    dir: Path
    model: torch.nn.Module
    history: list[dict] = field(default_factory=list)
    sample: str = ""

    @property
    def final(self) -> dict:
        return self.history[-1]

    def __repr__(self):
        f = self.final
        return f"Run({self.name}: train {f['train_loss']:.4f}, val {f['val_loss']:.4f}, {f['elapsed_s']:.0f}s)"


@torch.no_grad()
def estimate_loss(model, ds: Dataset, batch_size: int, block_size: int, eval_batches: int, device: str) -> dict:
    model.eval()
    out = {}
    for split, data in [("train", ds.train), ("val", ds.val)]:
        losses = [model(*get_batch(data, batch_size, block_size, device))[1].item() for _ in range(eval_batches)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def generate_text(model, tok: CharTokenizer, prompt: str = "\n", max_new_tokens: int = 300, **kw) -> str:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    idx = torch.tensor([tok.encode(prompt)], device=device)
    out = tok.decode(model.generate(idx, max_new_tokens, **kw)[0].tolist())
    model.train(was_training)
    return out


def train(model, ds: Dataset, cfg: TrainConfig | None = None, verbose: bool = True, **overrides) -> Run:
    """Treina `model` em `ds` com AdamW e lr constante. Aceita uma TrainConfig e/ou campos avulsos (steps=..., lr=...)."""
    cfg = replace(cfg or TrainConfig(), **overrides)
    steps = min(cfg.steps, int(os.environ["LAB_MAX_STEPS"])) if "LAB_MAX_STEPS" in os.environ else cfg.steps
    device = cfg.device or pick_device()
    block_size = cfg.block_size or model.block_size or 8
    model.to(device).train()
    torch.manual_seed(cfg.seed)  # mesma sequência de batches entre execuções comparadas
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    run_dir = RUNS_DIR / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        **asdict(cfg), "steps": steps, "block_size": block_size, "device": device, "dataset": ds.name,
        "model": type(model).__name__, "n_params": count_params(model),
        "model_config": model.config_dict() if hasattr(model, "config_dict") else None,
    }
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if verbose:
        print(f"{cfg.name}: {meta['n_params']:,} parâmetros, {steps} passos em {device}")

    run = Run(cfg.name, run_dir, model)
    t0 = time.time()
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "elapsed_s"])
        for step in range(steps + 1):
            if step % cfg.eval_every == 0 or step == steps:
                losses = estimate_loss(model, ds, cfg.batch_size, block_size, cfg.eval_batches, device)
                row = {"step": step, "train_loss": losses["train"], "val_loss": losses["val"], "elapsed_s": time.time() - t0}
                run.history.append(row)
                writer.writerow([step, f"{row['train_loss']:.4f}", f"{row['val_loss']:.4f}", f"{row['elapsed_s']:.1f}"])
                f.flush()
                if verbose:
                    print(f"  passo {step:5d} | treino {row['train_loss']:.4f} | validação {row['val_loss']:.4f} | {row['elapsed_s']:.0f}s")
            if step == steps:
                break
            _, loss = model(*get_batch(ds.train, cfg.batch_size, block_size, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

    if cfg.sample_tokens:
        run.sample = generate_text(model, ds.tok, cfg.sample_prompt, cfg.sample_tokens)
        (run_dir / "sample.txt").write_text(run.sample)
    if cfg.save:
        save_checkpoint(model, ds.tok, run_dir / "model.pt")
    return run


def save_checkpoint(model: GPT, tok: CharTokenizer, path: Path):
    torch.save({"model_config": model.config_dict(), "state_dict": model.state_dict(), "tokenizer": tok.to_json()}, path)


def load_checkpoint(path, device: str | None = None) -> tuple[GPT, CharTokenizer]:
    """Carrega um GPT salvo com save=True. Aceita o caminho do arquivo ou o nome da execução."""
    path = Path(path)
    if not path.suffix:
        path = RUNS_DIR / path / "model.pt"
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device or pick_device()).eval(), CharTokenizer.from_json(ckpt["tokenizer"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="shakespeare", help="shakespeare ou tinystories")
    parser.add_argument("--max_mb", type=int, default=50, help="MB do TinyStories usados no treino")
    model_fields = {f.name: f.default for f in fields(GPTConfig) if f.name not in ("vocab_size", "block_size")}
    train_fields = {f.name: f.default for f in fields(TrainConfig) if f.name not in ("block_size", "device", "save")}
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", action="store_true")
    for name, default in {**model_fields, **train_fields}.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{name}", type=lambda s: s.lower() in ("1", "true", "sim", "yes"), default=default)
        else:
            parser.add_argument(f"--{name}", type=type(default), default=default)
    args = vars(parser.parse_args())

    ds = shakespeare() if args["dataset"] == "shakespeare" else tinystories(args["max_mb"])
    print(ds)
    mcfg = GPTConfig(vocab_size=ds.tok.vocab_size, block_size=args["block_size"], **{k: args[k] for k in model_fields})
    tcfg = TrainConfig(device=args["device"], save=args["save"], **{k: args[k] for k in train_fields})
    seed_everything(tcfg.seed)
    run = train(GPT(mcfg), ds, tcfg)
    print("\n--- amostra ---\n" + run.sample)


if __name__ == "__main__":
    main()
