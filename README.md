# model-training

Laboratório para aprender os componentes de modelos de linguagem: um GPT pequeno, treinado no Mac,
que vai sendo desmontado e modificado experimento a experimento.

## Ambiente

Requer macOS com Apple Silicon e [Miniconda](https://docs.anaconda.com/miniconda/).

```bash
conda env create -f environment.yml   # cria o ambiente "model-training" e instala o projeto
conda activate model-training
```

As dependências ficam em `pyproject.toml` (PyTorch com backend MPS, MLX, `mlx-lm`, NumPy, Matplotlib).
Depois de mudar a lista, rode `pip install -e .` com o ambiente ativo.

Para conferir os backends:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
python -m lab.check_mlx
```

## Rodando um experimento

```bash
python -m lab.train --name base               # ~1 min num M-series, ~0,8M parâmetros
python -m lab.train --name lr3e-3 --lr 3e-3   # qualquer campo de Config vira flag
python -m lab.plot                            # compara a loss de validação de todas as execuções
```

Cada execução grava em `runs/<nome>/`:

| Arquivo | Conteúdo |
|---|---|
| `config.json` | hiperparâmetros, dispositivo e número de parâmetros |
| `metrics.csv` | `step, train_loss, val_loss, elapsed_s` |
| `sample.txt` | texto gerado no fim do treino |

`runs/comparacao.png` mostra as curvas lado a lado.

## Estrutura

```
lab/
  data.py       Tiny Shakespeare + tokenizador por caractere
  model.py      GPT mínimo (placeholder da Fase 0; reescrito peça por peça na Fase 1)
  train.py      loop de treino, registro de loss e geração
  plot.py       comparação de curvas
  check_mlx.py  teste rápido do MLX na GPU
```
