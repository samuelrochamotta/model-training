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

## Fase 1: os notebooks

O GPT é construído uma peça por vez em `notebooks/`, em ordem. Cada notebook escreve a peça nova à mão,
testa, treina um modelo de referência e traz células de experimento para variar parâmetros.

```bash
conda activate model-training
pip install -e .        # uma vez, para instalar o Jupyter (ver pyproject.toml)
jupyter lab notebooks/  # ou abra a pasta no VS Code e escolha o kernel do ambiente model-training
```

| Notebook | Peça nova | Experimentos | Diário |
|---|---|---|---|
| `01_bigramas` | tabela "caractere atual → próximo" | learning rate, block_size, temperatura, CPU vs. MPS | F1-BIG |
| `02_atencao` | uma cabeça de self-attention + embeddings de posição | contexto, sem posição, sem máscara, head_size | F1-ATN |
| `03_varias_cabecas` | várias cabeças + projeção | número de cabeças, ablação de cabeças | F1-MHA |
| `04_mlp` | MLP por posição | largura, ativação, MLP sem atenção | F1-MLP |
| `05_residual` | blocos empilhados + residual | profundidade, gradiente por camada, escala do fluxo | F1-RES |
| `06_normalizacao` | LayerNorm (pre e post) | posição da norma, lr alto, 8 blocos | F1-NRM |
| `07_shakespeare` | GPT completo, atenção em lote, dropout | overfitting, temperatura e top-k, mapas de atenção, GPT de 10M | F1-SHK |
| `08_tinystories` | dados maiores | 1 MB vs. 50 MB, tamanho do modelo | F1-TS |

Os notebooks 01 a 06 usam modelos minúsculos (cada treino leva cerca de um minuto). O 07 e o 08 têm
treinos longos, que também podem rodar no terminal com `python -m lab.train` (o comando exato está no notebook).

Para conferir rapidamente que tudo roda, limite todos os treinos a poucos passos:

```bash
LAB_MAX_STEPS=30 jupyter nbconvert --to notebook --execute --output-dir /tmp notebooks/*.ipynb
pytest -q   # testes das peças de referência (formatos, máscara causal, atenção em lote)
```

## Rodando um experimento pela linha de comando

```bash
python -m lab.train --name base                                   # GPT pequeno no Shakespeare
python -m lab.train --name lr3e-3 --lr 3e-3                       # qualquer campo de GPTConfig ou TrainConfig vira flag
python -m lab.train --name ts --dataset tinystories --max_mb 50   # TinyStories (baixa na primeira vez)
python -m lab.plot                                                # compara a loss de validação de todas as execuções
```

Cada execução grava em `runs/<nome>/`:

| Arquivo | Conteúdo |
|---|---|
| `config.json` | hiperparâmetros, configuração do modelo, dispositivo e número de parâmetros |
| `metrics.csv` | `step, train_loss, val_loss, elapsed_s` |
| `sample.txt` | texto gerado no fim do treino |
| `model.pt` | pesos, configuração e tokenizer (só com `--save` ou `save=True`) |

## Estrutura

```
lab/
  data.py       Tiny Shakespeare e TinyStories, tokenizador por caractere, get_batch
  model.py      versão de referência de cada peça (bigramas → GPT completo, configurável por GPTConfig)
  train.py      loop de treino usado pelos notebooks e pela linha de comando; checkpoints
  plot.py       comparação de curvas, tabela de resultados, mapa de atenção
  check_mlx.py  teste rápido do MLX na GPU
notebooks/      Fase 1, um notebook por peça
tests/          testes das peças de referência
```
