"""GGUF files read as stored bytes: metadata, tensor infos and raw block rows.

``sources/gguf.py`` dequantizes through the ``gguf`` package, which re-quantizes nothing but also
cannot hand out the stored blocks (and does not know Q2_0). A bit-exact import needs the bytes
themselves, so this reader parses the GGUF v3 container directly (header, key/values, tensor
infos, aligned data section) and memory-maps the files. Several shards of one split model are one
logical store keyed by tensor name.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
import struct
from typing import Iterable

import numpy as np

from tools.artifact.codecs.ggml import block_geometry

_MAGIC = b"GGUF"
# GGUF metadata value types.
_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_STRING, _ARRAY = 8, 9


@dataclass(frozen=True, slots=True)
class GgufTensor:
    name: str
    dims: tuple[int, ...]  # ggml order: dims[0] is the fastest (the row length K)
    ggml_type: int
    file: Path
    offset: int            # absolute byte offset in `file`
    nbytes: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Row-major shape (slowest first), as PyTorch/Safetensors order it."""

        return tuple(reversed(self.dims))

    @property
    def row_elems(self) -> int:
        return self.dims[0]

    @property
    def rows(self) -> int:
        return prod(self.dims[1:]) if len(self.dims) > 1 else 1

    @property
    def row_bytes(self) -> int:
        elems, size = block_geometry(self.ggml_type)
        if self.row_elems % elems:
            raise ValueError(f"{self.name}: row of {self.row_elems} is not whole blocks of {elems}")
        return self.row_elems // elems * size


class _Cursor:
    def __init__(self, data: memoryview):
        self.data = data
        self.at = 0

    def take(self, fmt: str):
        size = struct.calcsize(fmt)
        value = struct.unpack_from(fmt, self.data, self.at)
        self.at += size
        return value[0] if len(value) == 1 else value

    def string(self) -> str:
        length = self.take("<Q")
        value = bytes(self.data[self.at : self.at + length]).decode("utf-8")
        self.at += length
        return value

    def value(self, kind: int):
        if kind in _SCALARS:
            return self.take(_SCALARS[kind])
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            inner = self.take("<I")
            count = self.take("<Q")
            if inner in _SCALARS and inner != 7:
                fmt = _SCALARS[inner]
                size = struct.calcsize(fmt)
                values = np.frombuffer(self.data, dtype=np.dtype(fmt), count=count, offset=self.at)
                self.at += size * count
                return values.tolist()
            return [self.value(inner) for _ in range(count)]
        raise ValueError(f"unknown GGUF metadata type {kind}")


def _parse(path: Path) -> tuple[dict, list[GgufTensor]]:
    with path.open("rb") as stream:
        head = stream.read(1 << 26)  # metadata of these files is a few MB; tokenizer arrays included
    cursor = _Cursor(memoryview(head))
    if cursor.take("4s") != _MAGIC:
        raise ValueError(f"{path}: not a GGUF file")
    version = cursor.take("<I")
    if version != 3:
        raise ValueError(f"{path}: GGUF version {version} is not supported (3 required)")
    count, kvs = cursor.take("<Q"), cursor.take("<Q")
    metadata = {}
    for _ in range(kvs):
        key = cursor.string()
        metadata[key] = cursor.value(cursor.take("<I"))
    infos = []
    for _ in range(count):
        name = cursor.string()
        ndims = cursor.take("<I")
        dims = tuple(cursor.take("<Q") for _ in range(ndims))
        kind = cursor.take("<I")
        offset = cursor.take("<Q")
        infos.append((name, dims, kind, offset))
    alignment = int(metadata.get("general.alignment", 32))
    data_start = (cursor.at + alignment - 1) // alignment * alignment
    tensors = []
    for name, dims, kind, offset in infos:
        elems, size = block_geometry(kind)
        total = prod(dims)
        if total % elems:
            raise ValueError(f"{path}: {name} is not whole blocks")
        tensors.append(GgufTensor(name, dims, kind, path, data_start + offset, total // elems * size))
    return metadata, tensors


class GgufStore:
    """The tensors of one or more GGUF shards; raw rows are read from memory maps."""

    def __init__(self, paths: Iterable[str | Path]):
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("a GGUF store needs at least one file")
        self.metadata: dict = {}
        self.tensors: dict[str, GgufTensor] = {}
        for path in self.paths:
            metadata, tensors = _parse(path)
            for key, value in metadata.items():
                if key.startswith("split.") or key == "GGUF.version":
                    continue
                if key in self.metadata and self.metadata[key] != value:
                    raise ValueError(f"{path}: metadata {key} differs between shards")
                self.metadata[key] = value
            for tensor in tensors:
                if tensor.name in self.tensors:
                    raise ValueError(f"{path}: duplicate tensor {tensor.name}")
                self.tensors[tensor.name] = tensor
        self._maps: dict[Path, np.memmap] = {}
        self.bytes_read = 0

    @property
    def path(self) -> Path:
        return self.paths[0]

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def tensor(self, name: str) -> GgufTensor:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise ValueError(f"GGUF store has no tensor {name!r}") from error

    def _map(self, path: Path) -> np.memmap:
        if path not in self._maps:
            self._maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._maps[path]

    def raw(self, name: str, row_begin: int = 0, row_end: int | None = None) -> np.ndarray:
        """Stored bytes of rows [row_begin, row_end) as uint8 [rows, row_bytes] (a read-only view)."""

        t = self.tensor(name)
        row_end = t.rows if row_end is None else row_end
        if not 0 <= row_begin <= row_end <= t.rows:
            raise ValueError(f"{name}: rows [{row_begin},{row_end}) exceed {t.rows}")
        first = t.offset + row_begin * t.row_bytes
        data = self._map(t.file)[first : first + (row_end - row_begin) * t.row_bytes]
        self.bytes_read += data.size
        return data.reshape(row_end - row_begin, t.row_bytes)

    def plain(self, name: str) -> np.ndarray:
        """An F32, F16 or BF16 tensor's values as float64 in row-major shape (exact widening)."""

        t = self.tensor(name)
        raw = np.ascontiguousarray(self.raw(name)).reshape(-1)
        if t.ggml_type == 0:
            values = raw.view("<f4").astype(np.float64)
        elif t.ggml_type == 1:
            values = raw.view("<f2").astype(np.float64)
        elif t.ggml_type == 30:
            values = (raw.view("<u2").astype(np.uint32) << 16).view("<f4").astype(np.float64)
        else:
            raise ValueError(f"{name}: type {t.ggml_type} is not a plain float tensor")
        return values.reshape(t.shape)

    def close(self) -> None:
        self._maps.clear()

    def __enter__(self) -> "GgufStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
