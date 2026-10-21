"""Compara curvas de loss de várias execuções.

Nos notebooks:
    compare(["01-bigram", "02-atencao"])            # validação (linha cheia) e treino (tracejada)
    compare(prefix="03-cabecas")                    # todas as execuções cujo nome começa assim

Na linha de comando:
    python -m lab.plot            # todas as execuções em runs/
    python -m lab.plot base lr3e-3
"""

import csv
import json
import sys

import matplotlib.pyplot as plt

from lab.train import RUNS_DIR


def load_metrics(name: str) -> list[dict]:
    with open(RUNS_DIR / name / "metrics.csv") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def all_runs(prefix: str = "") -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted(p.name for p in RUNS_DIR.iterdir() if (p / "metrics.csv").exists() and p.name.startswith(prefix))


def compare(names: list[str] | None = None, prefix: str = "", train: bool = True, x: str = "step",
            title: str | None = None, ylim: tuple | None = None, ax=None):
    """Plota a loss de validação (cheia) e, com train=True, a de treino (tracejada) de cada execução.
    x="elapsed_s" troca o eixo de passos por segundos."""
    names = names or all_runs(prefix)
    missing = [n for n in names if not (RUNS_DIR / n / "metrics.csv").exists()]
    if missing:
        print("sem dados (rode o notebook correspondente antes):", ", ".join(missing))
    names = [n for n in names if n not in missing]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    for name in names:
        rows = load_metrics(name)
        xs = [r[x] for r in rows]
        (line,) = ax.plot(xs, [r["val_loss"] for r in rows], label=name)
        if train:
            ax.plot(xs, [r["train_loss"] for r in rows], ls="--", alpha=0.5, color=line.get_color())
    ax.set_xlabel("passo" if x == "step" else "segundos")
    ax.set_ylabel("loss (cheia: validação, tracejada: treino)" if train else "loss de validação")
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def table(names: list[str] | None = None, prefix: str = ""):
    """Resumo em texto: parâmetros, loss final de treino e validação, tempo."""
    lines = [f"{'execução':<32} {'parâmetros':>11} {'treino':>7} {'valid.':>7} {'tempo':>7}"]
    for name in names or all_runs(prefix):
        if not (RUNS_DIR / name / "metrics.csv").exists():
            continue
        last = load_metrics(name)[-1]
        n = json.loads((RUNS_DIR / name / "config.json").read_text()).get("n_params", 0)
        lines.append(f"{name:<32} {n:>11,} {last['train_loss']:>7.4f} {last['val_loss']:>7.4f} {last['elapsed_s']:>6.0f}s")
    print("\n".join(lines))


def plot_attention(att, labels: list[str], ax=None, title: str | None = None):
    """Mapa de calor de uma matriz de atenção (T, T): linha = quem olha, coluna = para quem olha."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(att, cmap="Blues", vmin=0, vmax=1)
    shown = [repr(c)[1:-1] for c in labels]
    ax.set_xticks(range(len(labels)), shown, fontsize=7)
    ax.set_yticks(range(len(labels)), shown, fontsize=7)
    ax.set_xlabel("posição observada (key)")
    ax.set_ylabel("posição que pergunta (query)")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def main():
    names = sys.argv[1:] or all_runs()
    ax = compare(names, train=False)
    out = RUNS_DIR / "comparacao.png"
    ax.figure.savefig(out, dpi=120, bbox_inches="tight")
    print(f"gráfico salvo em {out}")


if __name__ == "__main__":
    main()
