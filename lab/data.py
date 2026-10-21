"""Dados da Fase 1: Tiny Shakespeare (~1 MB) e um recorte do TinyStories, ambos por caractere.

Uso nos notebooks:
    ds = shakespeare()             # Dataset com .tok, .train, .val
    x, y = get_batch(ds.train, batch_size=32, block_size=64)
"""

from dataclasses import dataclass
from pathlib import Path
import json
import urllib.request

import numpy as np
import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# No TinyStories, cada história termina com <|endoftext|>. Viramos isso num caractere só,
# o ASCII 4 ("End Of Transmission"), para o modelo aprender onde uma história acaba.
EOT = "\x04"


def load_text() -> str:
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    def __init__(self, text: str = "", chars: list[str] | None = None):
        self.chars = chars if chars is not None else sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    def encode_tensor(self, s: str) -> torch.Tensor:
        """Como encode, mas rápido para textos grandes (só caracteres ASCII), em uint8."""
        lut = np.zeros(256, dtype=np.uint8)
        for c, i in self.stoi.items():
            lut[ord(c)] = i
        return torch.from_numpy(lut[np.frombuffer(s.encode("ascii"), dtype=np.uint8)].copy())

    def to_json(self) -> str:
        return json.dumps(self.chars)

    @classmethod
    def from_json(cls, s: str) -> "CharTokenizer":
        return cls(chars=json.loads(s))


def train_val_split(text: str, tok: CharTokenizer, val_frac: float = 0.1):
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(data) * (1 - val_frac))
    return data[:n], data[n:]


@dataclass
class Dataset:
    name: str
    tok: CharTokenizer
    train: torch.Tensor
    val: torch.Tensor

    def __repr__(self):
        return (f"Dataset({self.name}: vocabulário de {self.tok.vocab_size} caracteres, "
                f"{len(self.train):,} tokens de treino, {len(self.val):,} de validação)")


def shakespeare() -> Dataset:
    text = load_text()
    tok = CharTokenizer(text)
    train, val = train_val_split(text, tok)
    return Dataset("shakespeare", tok, train, val)


def _download_prefix(url: str, path: Path, max_bytes: int | None):
    """Baixa só os primeiros max_bytes do arquivo (o de treino tem 2,2 GB)."""
    req = urllib.request.Request(url)
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        got = 0
        while chunk := r.read(1 << 20):
            if max_bytes and got + len(chunk) > max_bytes:
                chunk = chunk[: max_bytes - got]
            f.write(chunk)
            got += len(chunk)
            if max_bytes and got >= max_bytes:
                break
    tmp.rename(path)


_ASCII_FIXES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " "})


def clean_stories(raw: str) -> str:
    """Normaliza pontuação, descarta caracteres fora do ASCII e marca o fim de cada história com EOT."""
    stories = [s.strip() for s in raw.split("<|endoftext|>")]
    stories = [s for s in stories if s]
    text = EOT.join(stories) + EOT
    text = text.translate(_ASCII_FIXES)
    return text.encode("ascii", errors="ignore").decode("ascii")


def tinystories(max_mb: int = 50) -> Dataset:
    """Os primeiros `max_mb` MB do TinyStories (versão GPT-4) para treino e o arquivo de validação inteiro (~22 MB)."""
    DATA_DIR.mkdir(exist_ok=True)
    texts = {}
    for split, limit in [("train", max_mb * 1_000_000), ("valid", None)]:
        raw_path = DATA_DIR / f"tinystories_{split}_{max_mb}mb.txt" if limit else DATA_DIR / f"tinystories_{split}.txt"
        if not raw_path.exists():
            print(f"baixando TinyStories ({split})...")
            _download_prefix(TINYSTORIES_URL.format(split=split), raw_path, limit)
        raw = raw_path.read_text(encoding="utf-8", errors="ignore")
        if limit:  # a última história foi cortada no meio pelo limite de bytes
            raw = raw[: raw.rfind("<|endoftext|>")]
        texts[split] = clean_stories(raw)
    tok = CharTokenizer(texts["train"] + texts["valid"])
    return Dataset(f"tinystories-{max_mb}mb", tok, tok.encode_tensor(texts["train"]), tok.encode_tensor(texts["valid"]))


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str = "cpu"):
    """Sorteia batch_size janelas de block_size tokens. y é x deslocado de uma posição: o próximo token."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).long()
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).long()
    return x.to(device), y.to(device)
