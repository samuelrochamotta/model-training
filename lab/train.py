"""Treina o GPT mínimo, registra a loss em runs/<nome>/metrics.csv e gera texto no fim.

Uso:
    python -m lab.train --name base
    python -m lab.train --name lr3e-3 --lr 3e-3
"""

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from lab.data import CharTokenizer, load_text, train_val_split
from lab.model import GPT

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class Config:
    name: str = "base"
    seed: int = 1337
    block_size: int = 128
    batch_size: int = 32
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    max_steps: int = 2000
    eval_every: int = 100
    eval_batches: int = 20
    sample_tokens: int = 500


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(data, cfg, device):
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + cfg.block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, cfg, device))[1].item() for _ in range(cfg.eval_batches)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    for field, default in asdict(Config()).items():
        parser.add_argument(f"--{field}", type=type(default), default=default)
    cfg = Config(**vars(parser.parse_args()))

    torch.manual_seed(cfg.seed)
    device = pick_device()
    text = load_text()
    tok = CharTokenizer(text)
    train_data, val_data = train_val_split(text, tok)

    model = GPT(tok.vocab_size, cfg.block_size, cfg.d_model, cfg.n_layer, cfg.n_head, cfg.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    run_dir = RUNS_DIR / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "device": device, "n_params": n_params}, indent=2)
    )
    print(f"{cfg.name}: {n_params / 1e6:.2f}M parâmetros em {device}")

    t0 = time.time()
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "elapsed_s"])
        for step in range(cfg.max_steps + 1):
            if step % cfg.eval_every == 0:
                losses = estimate_loss(model, {"train": train_data, "val": val_data}, cfg, device)
                elapsed = time.time() - t0
                writer.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{elapsed:.1f}"])
                f.flush()
                print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.0f}s")
            if step == cfg.max_steps:
                break
            _, loss = model(*get_batch(train_data, cfg, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    start = torch.tensor([tok.encode("\n")], device=device)
    sample = tok.decode(model.generate(start, cfg.sample_tokens)[0].tolist())
    (run_dir / "sample.txt").write_text(sample)
    print("\n--- amostra ---\n" + sample)


if __name__ == "__main__":
    main()
