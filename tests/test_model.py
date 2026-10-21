"""Testes das peças de referência: formatos, causalidade e equivalência entre as duas atenções.

    pytest -q
"""

import math

import pytest
import torch

from lab.data import CharTokenizer, clean_stories, EOT
from lab.model import (BigramLM, CausalSelfAttention, FeedForward, GPT, GPTConfig, Head,
                       MultiHeadAttention, count_params)

V, T, B = 20, 16, 3


def test_bigram_loss_inicial_perto_de_ln_V():
    torch.manual_seed(0)
    m = BigramLM(V)
    torch.nn.init.zeros_(m.table.weight)  # logits todos iguais -> previsão uniforme
    _, loss = m(torch.randint(V, (B, T)), torch.randint(V, (B, T)))
    assert loss.item() == pytest.approx(math.log(V), rel=1e-5)


@pytest.mark.parametrize("make", [
    lambda: Head(32, 8, T),
    lambda: MultiHeadAttention(32, 4, T),
    lambda: CausalSelfAttention(32, 4, T),
])
def test_atencao_causal_nao_ve_o_futuro(make):
    torch.manual_seed(0)
    layer = make()
    x = torch.randn(1, T, 32)
    y1 = layer(x)
    x2 = x.clone()
    x2[:, 10:] = torch.randn(1, T - 10, 32)  # muda só o "futuro" a partir da posição 10
    y2 = layer(x2)
    assert torch.allclose(y1[:, :10], y2[:, :10], atol=1e-6)
    assert not torch.allclose(y1[:, 10:], y2[:, 10:])


def test_cabecas_em_lote_igual_a_cabecas_separadas():
    torch.manual_seed(0)
    d, nh = 32, 4
    mha, csa = MultiHeadAttention(d, nh, T), CausalSelfAttention(d, nh, T)
    with torch.no_grad():
        q = torch.cat([h.query.weight for h in mha.heads])
        k = torch.cat([h.key.weight for h in mha.heads])
        v = torch.cat([h.value.weight for h in mha.heads])
        csa.qkv.weight.copy_(torch.cat([q, k, v]))
        csa.proj.load_state_dict(mha.proj.state_dict())
    x = torch.randn(B, T, d)
    assert torch.allclose(mha(x), csa(x), atol=1e-5)
    assert count_params(mha) == count_params(csa)


def test_feedforward_preserva_formato():
    assert FeedForward(32, 4)(torch.randn(B, T, 32)).shape == (B, T, 32)


@pytest.mark.parametrize("kw", [
    {}, {"norm": "none"}, {"norm": "post"}, {"residual": False}, {"mlp": False},
    {"attn": "batched", "n_layer": 2, "n_head": 4}, {"pos_emb": False}, {"dropout": 0.1},
])
def test_gpt_variacoes(kw):
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=V, block_size=T, d_model=32, **kw))
    logits, loss = m(torch.randint(V, (B, T)), torch.randint(V, (B, T)))
    assert logits.shape == (B, T, V) and torch.isfinite(loss)
    assert m.generate(torch.zeros(1, 1, dtype=torch.long), 2 * T).shape == (1, 2 * T + 1)


def test_gpt_loss_inicial_perto_de_ln_V():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=V, block_size=T, d_model=32, n_layer=2, n_head=2))
    _, loss = m(torch.randint(V, (8, T)), torch.randint(V, (8, T)))
    assert abs(loss.item() - math.log(V)) < 0.1


def test_tokenizer_e_limpeza_tinystories():
    text = clean_stories("Once upon a time “hi”.<|endoftext|>\nThe end — ok…<|endoftext|>")
    assert text == 'Once upon a time "hi".' + EOT + "The end - ok..." + EOT
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text
    assert tok.encode_tensor(text).tolist() == tok.encode(text)
    assert CharTokenizer.from_json(tok.to_json()).chars == tok.chars
