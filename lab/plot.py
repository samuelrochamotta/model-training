"""Compara curvas de loss de validação de várias execuções.

Uso:
    python -m lab.plot            # todas as execuções em runs/
    python -m lab.plot base lr3e-3
"""

import csv
import sys

import matplotlib.pyplot as plt

from lab.train import RUNS_DIR


def main():
    names = sys.argv[1:] or sorted(p.name for p in RUNS_DIR.iterdir() if (p / "metrics.csv").exists())
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        with open(RUNS_DIR / name / "metrics.csv") as f:
            rows = list(csv.DictReader(f))
        ax.plot([int(r["step"]) for r in rows], [float(r["val_loss"]) for r in rows], label=name)
    ax.set_xlabel("passo")
    ax.set_ylabel("loss de validação")
    ax.legend()
    ax.grid(alpha=0.3)
    out = RUNS_DIR / "comparacao.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"gráfico salvo em {out}")


if __name__ == "__main__":
    main()
