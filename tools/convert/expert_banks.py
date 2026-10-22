"""Exact NVFP4 routed-expert banks shared by the Qwen3.5 and Qwen4Exp adapters.

A bank ``[experts, hidden, intermediate]`` imports each expert's gate, up and down NVFP4 matrices
bit-exactly into ``nvfp4_expert_rg16_v1`` records (design docs/maintainer/qwen3_8-flash-next-design.md
§6.2). The sources are ModelOpt exports (``weight``, ``weight_scale``, ``weight_scale_2`` and, for a
W4A4 bank, ``input_scale``) or weight-only compressed-tensors exports (``weight_packed``,
``weight_global_scale``). A *single* bank is one gated expert stored as a one-record bank: a
Qwen3.5 shared expert, which the routed-expert kernels run as a resident job.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.artifact.codecs.nvfp4_expert import encode_nvfp4_expert_bank
from tools.artifact.layouts import expert_bank_geometry

from .methods import PrepareRequest, PreparedMethod
from .sources.modelopt import nvfp4_matrix_words, nvfp4_packed_matrix_words
from .sources.safetensors import SafetensorsSource


@dataclass(frozen=True, slots=True)
class ExpertBankSource:
    """One layer's NVFP4 experts; only ``import_expert_bank`` can read it. ``packed`` selects the
    weight-only compressed-tensors export (``weight_packed`` / ``weight_global_scale``, no activation
    scales: a W4A16 bank) instead of ModelOpt's. A ModelOpt bank is W4A4 when its matrices carry
    ``input_scale`` and W4A16 when they do not (``weight_only``)."""

    shape: tuple[int, int, int]
    label: str
    store: SafetensorsSource
    prefix: str  # "...mlp.experts." (expert e's gate is f"{prefix}{e}.gate_proj"), or a single leaf base
    packed: bool = False
    single: bool = False  # "...mlp.shared_expert.": the one expert's gate is f"{prefix}gate_proj"
    read_encoded: object = None
    weight_divisor: object = None
    input_divisor: object = None
    input_scale: object = None

    def leaf(self, expert: int, role: str) -> str:
        if self.single:
            if expert != 0:
                raise IndexError(f"{self.label}: a single bank has one expert")
            return f"{self.prefix}{role}_proj"
        return f"{self.prefix}{expert}.{role}_proj"

    @property
    def weight_only(self) -> bool:
        """Whether the bank has no activation scales (a W4A16 bank)."""
        return self.packed or not self.store.has(self.leaf(0, "gate") + ".input_scale")

    def values(self, begin: int = 0, end: int | None = None) -> torch.Tensor:
        raise ValueError(f"{self.label}: an expert bank is imported exactly, never as values")


def weight_only_bank(source) -> bool:
    """Whether a routed-expert parameter's source is a weight-only compressed-tensors (W4A16) bank."""

    return isinstance(source, ExpertBankSource) and source.packed


_BANK_CHUNK = 32  # experts encoded per step; bounds conversion memory at ~0.4 GB (Qwen4Exp records)


def import_expert_bank(request: PrepareRequest) -> PreparedMethod:
    """Write one layer's experts bit-exactly in ``nvfp4_expert_rg16_v1`` (design §6.2)."""

    if request.target.format != "nvfp4_mul" or request.target.layout != "nvfp4_expert_rg16_v1":
        raise ValueError("import_expert_bank requires nvfp4_mul in nvfp4_expert_rg16_v1")
    if request.parameters:
        raise ValueError("import_expert_bank accepts no numerical parameters")
    if len(request.inputs) != 1 or not isinstance(request.inputs[0].source, ExpertBankSource):
        raise ValueError("import_expert_bank requires exactly one expert-bank source")
    source = request.inputs[0].source
    if tuple(request.target.shape) != source.shape:
        raise ValueError("expert bank target shape differs from its source")
    experts, hidden, intermediate = source.shape
    geometry = expert_bank_geometry("nvfp4_mul", source.shape)
    words = nvfp4_packed_matrix_words if source.packed else nvfp4_matrix_words
    # Bound the in-flight codes at ~0.4 GB whatever the record size.
    chunk = max(1, _BANK_CHUNK * 2_764_800 // geometry.record_bytes)

    def produce(output):
        multipliers = []
        for first in range(0, experts, chunk):
            count = min(chunk, experts - first)
            gc, gs, uc, us, dc, ds, mult = [], [], [], [], [], [], []
            for e in range(first, first + count):
                gate = words(source.store, source.leaf(e, "gate"), (intermediate, hidden))
                up = words(source.store, source.leaf(e, "up"), (intermediate, hidden))
                down = words(source.store, source.leaf(e, "down"), (hidden, intermediate))
                gc.append(gate.codes)
                gs.append(gate.scales)
                uc.append(up.codes)
                us.append(up.scales)
                dc.append(down.codes)
                ds.append(down.scales)
                mult.append(torch.stack((gate.weight_scale_2, up.weight_scale_2, down.weight_scale_2)))
            part = encode_nvfp4_expert_bank(
                torch.stack(gc), torch.stack(gs), torch.stack(uc), torch.stack(us),
                torch.stack(dc), torch.stack(ds), torch.stack(mult), (count, hidden, intermediate),
            )
            output.write_bytes(first * geometry.record_stride, memoryview(part)[: count * geometry.record_stride])
            multipliers.append(torch.stack(mult))
        plane = torch.cat(multipliers).contiguous()
        output.write_bytes(geometry.multiplier_offset, plane.numpy().tobytes())

    return request.job(produce=produce)
