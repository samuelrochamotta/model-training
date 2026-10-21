"""Versão de referência das peças construídas nos notebooks da Fase 1.

Cada notebook escreve uma peça nova à mão e importa daqui as peças dos notebooks anteriores.
A ordem do arquivo segue a ordem dos notebooks:

    01 BigramLM            tabela "token atual -> próximo token"
    02 Head                uma cabeça de self-attention causal
    03 MultiHeadAttention  várias cabeças em paralelo + projeção
       CausalSelfAttention a mesma conta, com as cabeças num tensor só (mais rápida, notebook 07)
    04 FeedForward         MLP aplicada a cada posição
    05 Block               atenção + MLP, com ou sem conexões residuais
    06 (normalização)      opção norm= do Block: "none", "pre" ou "post"
    07 GPT / GPTConfig     o modelo completo, configurável peça por peça

Todos os modelos seguem o mesmo contrato: forward(idx, targets=None) -> (logits, loss) e generate().
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModel(nn.Module):
    """Base comum: calcula a loss e gera texto. As subclasses implementam logits(idx)."""

    block_size: int | None = None  # contexto máximo; None = o modelo só olha o último token

    def logits(self, idx):
        raise NotImplementedError

    def forward(self, idx, targets=None):
        logits = self.logits(idx)  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None,
                 stop_token: int | None = None):
        for _ in range(max_new_tokens):
            ctx = idx if self.block_size is None else idx[:, -self.block_size:]
            logits = self.logits(ctx)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_token is not None and (nxt == stop_token).all():
                break
        return idx


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- 01 · bigramas


class BigramLM(LanguageModel):
    """Cada token olha só para si mesmo: a linha i da tabela são os logits do token que vem depois de i."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def logits(self, idx):
        return self.table(idx)


# ---------------------------------------------------------------- 02 · uma cabeça de atenção


class Head(nn.Module):
    """Uma cabeça de self-attention. Cada posição faz uma pergunta (query), anuncia o que tem (key)
    e entrega um conteúdo (value); o peso de i em j é softmax(q_i · k_j / sqrt(d))."""

    def __init__(self, d_model: int, head_size: int, block_size: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.causal = causal
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)), persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.last_att = None  # pesos da última chamada, para visualizar

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)  # (B, T, hs)
        att = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5  # (B, T, T)
        if self.causal:
            att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        self.last_att = att.detach()
        return self.dropout(att) @ v  # (B, T, hs)


# ---------------------------------------------------------------- 03 · várias cabeças


class MultiHeadAttention(nn.Module):
    """n_head cabeças independentes, cada uma com d_model // n_head dimensões; as saídas são
    concatenadas e misturadas por uma projeção linear."""

    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        assert d_model % n_head == 0, "d_model precisa ser divisível por n_head"
        self.heads = nn.ModuleList(Head(d_model, d_model // n_head, block_size, dropout, causal) for _ in range(n_head))
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class CausalSelfAttention(nn.Module):
    """A mesma conta de MultiHeadAttention, mas com todas as cabeças num tensor (B, n_head, T, hs).
    Mesmos parâmetros, bem mais rápida na GPU. O notebook 07 confere que as duas dão o mesmo resultado."""

    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        assert d_model % n_head == 0, "d_model precisa ser divisível por n_head"
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.causal = causal
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)), persistent=False)
        self.att_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.last_att = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5  # (B, nh, T, T)
        if self.causal:
            att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        self.last_att = att.detach()
        y = (self.att_dropout(att) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


# ---------------------------------------------------------------- 04 · MLP


ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "identity": nn.Identity}


class FeedForward(nn.Module):
    """Depois de trocar informação entre posições (atenção), cada posição "pensa sozinha":
    expande para mult * d_model, aplica a não linearidade e volta para d_model."""

    def __init__(self, d_model: int, mult: int = 4, activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            ACTIVATIONS[activation](),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- 05 e 06 · bloco, residual e normalização


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64       # contexto máximo (tokens)
    d_model: int = 64          # largura do fluxo residual
    n_layer: int = 1           # quantos blocos empilhados
    n_head: int = 1
    attn: str = "heads"        # "heads" (03, uma cabeça por vez) ou "batched" (07, mais rápida)
    mlp: bool = True           # 04
    mlp_mult: int = 4
    activation: str = "relu"
    residual: bool = True      # 05
    norm: str = "pre"          # 06: "none", "pre" (GPT-2 em diante) ou "post" (Transformer original)
    dropout: float = 0.0
    pos_emb: bool = True       # 02: sem isso a atenção não sabe a ordem dos tokens
    causal: bool = True        # 02: sem a máscara o modelo enxerga o futuro


class Block(nn.Module):
    """Atenção (comunicação entre posições) seguida de MLP (computação em cada posição).
    Com residual, cada subcamada soma uma correção ao fluxo x em vez de substituí-lo."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        Attn = MultiHeadAttention if cfg.attn == "heads" else CausalSelfAttention
        self.attn = Attn(cfg.d_model, cfg.n_head, cfg.block_size, cfg.dropout, cfg.causal)
        self.mlp = FeedForward(cfg.d_model, cfg.mlp_mult, cfg.activation, cfg.dropout) if cfg.mlp else None
        self.residual = cfg.residual
        self.norm = cfg.norm
        ln = (lambda: nn.LayerNorm(cfg.d_model)) if cfg.norm != "none" else nn.Identity
        self.ln1, self.ln2 = ln(), ln()

    def _sublayer(self, x, f, ln):
        h = f(ln(x)) if self.norm == "pre" else f(x)
        x = x + h if self.residual else h
        return ln(x) if self.norm == "post" else x

    def forward(self, x):
        x = self._sublayer(x, self.attn, self.ln1)
        if self.mlp is not None:
            x = self._sublayer(x, self.mlp, self.ln2)
        return x


# ---------------------------------------------------------------- 07 · o GPT completo


class GPT(LanguageModel):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model) if cfg.pos_emb else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model) if cfg.norm == "pre" else nn.Identity()
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Pesos pequenos (desvio 0,02, como no GPT-2): no início o modelo prevê quase uniforme.
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def logits(self, idx):
        T = idx.size(1)
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def config_dict(self) -> dict:
        return asdict(self.cfg)
