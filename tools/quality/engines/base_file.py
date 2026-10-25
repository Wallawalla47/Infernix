"""The header of llama.cpp's `_logits_` base file (apps/certify/llama_kld.h): what an engine adapter needs to feed
its engine the base's tokens and to emit its logits rows in the base's order for infernix-certify llama-rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BaseHeader:
    context: int
    vocab: int
    chunks: int
    tokens: np.ndarray  # chunks x context, int32

    @property
    def first_row(self) -> int:
        return self.context // 2

    @property
    def rows_per_chunk(self) -> int:
        return self.context - 1 - self.context // 2

    def chunk(self, index: int) -> np.ndarray:
        return self.tokens[index]


def read_header(path: Path) -> BaseHeader:
    with Path(path).open("rb") as f:
        if f.read(8) != b"_logits_":
            raise ValueError(f"{path} is not a llama.cpp _logits_ file")
        context, vocab, chunks = (int(v) for v in np.frombuffer(f.read(12), dtype="<i4"))
        tokens = np.frombuffer(f.read(4 * context * chunks), dtype="<i4").reshape(chunks, context)
    return BaseHeader(context, vocab, chunks, tokens)
