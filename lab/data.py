"""Baixa e prepara o Tiny Shakespeare (~1 MB) com tokenização por caractere."""

from pathlib import Path
import urllib.request

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_text() -> str:
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)


def train_val_split(text: str, tok: CharTokenizer, val_frac: float = 0.1):
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(data) * (1 - val_frac))
    return data[:n], data[n:]
