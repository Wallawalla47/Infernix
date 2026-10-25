"""The 2-bit screens of a Qwen3.8-Flash-Next (Qwen4Exp) artifact's heads (``--screen-heads``).

A screened head (include/infernix/ops/screened_head.h) scores every row of a head from its 2-bit copy
and computes only the near-best rows with the head itself. Two copies, both ``q2_g64_fp16`` in
``row_split_k128_v1`` (2.25 bits a weight) with ``grouped_mse_norm``'s scales, fitted from the LM
head's own source values (the checkpoint's ``lm_head``):

- ``screen/output_head`` [vocab, hidden]: every row of ``text/output_head`` in token order, read by the
  ``screened-lm-head`` fast path (Qwen3.8-Flash-Next: 248,320 x 2,560, 178.8 MB);
- ``screen/proposal_head`` [rows, hidden]: the rows of ``proposal/head`` in its order (the frequency
  shortlist of ``--proposal``), read by the MTP drafter's screened draft head (131,072 x 2,560,
  94.4 MB). Each row is quantized alone, so these are the same bytes as those rows of the LM head's
  copy; a copy of their own keeps the draft screen one contiguous read of only the rows it drafts from.

They are extra parameters outside the logical model, added after every other parameter: the existing
tensors keep their bytes and their order, and an engine binds a copy only for the path that reads it.
"""

from __future__ import annotations

from .methods import grouped_mse_norm
from .model import Parameter
from .recipe import Recipe
from .sources.logical import gather_source

FORMAT = "q2_g64_fp16"
OUTPUT_HEAD = "screen/output_head"
PROPOSAL_HEAD = "screen/proposal_head"


def add_screen_heads(recipe: Recipe) -> None:
    """Adds the screens of the LM head and, when the recipe has one (``--proposal`` first), of the
    proposal head."""

    model = recipe.model
    if model.config.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("--screen-heads screens a Qwen3.8-Flash-Next (Qwen4Exp) artifact's heads")
    if OUTPUT_HEAD in model.parameters or PROPOSAL_HEAD in model.parameters:
        raise ValueError("the head screens are already configured")
    head = model.parameters.get("text/output_head")
    if head is None or len(head.shape) != 2:
        raise ValueError("--screen-heads needs the text component's output head")
    model.add(Parameter(OUTPUT_HEAD, head.shape, head.source, residency="screen"))
    recipe.add_parameter(OUTPUT_HEAD)
    recipe.assign(OUTPUT_HEAD, format=FORMAT, method=grouped_mse_norm)
    if "proposal/head" in model.parameters:
        rows = model.parameters["proposal/head"].shape[0]
        ids = model.parameters["proposal/token_ids"].source.values(0, rows)
        source = gather_source(head.source, ids)
        model.add(Parameter(PROPOSAL_HEAD, (rows, head.shape[1]), source, residency="screen"))
        recipe.add_parameter(PROPOSAL_HEAD)
        recipe.assign(PROPOSAL_HEAD, format=FORMAT, method=grouped_mse_norm)
