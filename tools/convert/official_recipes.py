"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from .methods import (
    cast_direct,
    fp8_row_maxabs,
    grouped_absmax,
    grouped_mse,
    import_encoded,
    nvfp4_mse,
)

Q2 = "q2_g64_fp16"
Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None, method=grouped_absmax):
    method = method if format in (Q2, Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


_DRAFT_BF16 = (
    "/moe/router",
    "/moe/shared_score",
    "/attention_conv/kernel_projection",
    "/mlp_conv/kernel_projection",
    "/candidate_selector/hidden_projection",
)


def _optional(model, recipe, *, method=grouped_absmax):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format, method=method)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(_DRAFT_BF16):
                continue
            _assign(recipe, name, Q8, method=method)
    _share_draft_context(model, recipe)


def _dflash_drafters(model, recipe, *, method=grouped_absmax):
    """DFlash/DFlash2 drafters as NInfer's official Qwen3.8-27B artifact stores its DFlash2: every
    projection ``q8_g32_fp16`` except the dynamic-convolution kernel projections and the selector's
    hidden projection, which stay BF16 with the norms, base kernels and codebooks; the context K/V
    roles share the query block's K/V words. A drafter only proposes tokens the model verifies, so
    its precision changes acceptance, never output."""
    for name, parameter in model.parameters.items():
        if parameter.projection and name.startswith(("dflash/", "dflash2/")) and not name.endswith(_DRAFT_BF16):
            _assign(recipe, name, Q8, method=method)
    _share_draft_context(model, recipe)


def _share_draft_context(model, recipe):
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dflash2_nvfp4_gate_up(model, recipe):
    """Store each DFlash2 layer's MLP gate/up parent as NVFP4 with 16-bit activations."""
    if "dflash2" not in model.components:
        return
    layers = model.components["dflash2"]["config"]["num_hidden_layers"]
    for layer in range(layers):
        recipe.assign(
            [f"dflash2/layers/{layer}/mlp/gate", f"dflash2/layers/{layer}/mlp/up"],
            format="nvfp4",
            method=nvfp4_mse,
            activation_policy="A16Only",
        )


def _dense_groupwise(model, recipe, vocabulary, gate_up=Q4, *, method=grouped_absmax):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe, method=method)
    _assign(recipe, "text/token_embedding", vocabulary, method=method)
    _assign(recipe, "text/output_head", vocabulary, method=method)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(("/mlp/gate", "/mlp/up")):
            format = gate_up
        elif name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format, method=method)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_8_27b_q6(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8, gate_up=Q6)


def qwen3_5_gguf_q8_0(model, recipe, sources):
    """A llama.cpp Q8_0 GGUF of a Qwen3.5 dense model, bit-exact (``tools.convert --gguf``,
    ``qwen3_5_gguf``): every Q8_0 matrix, the token embedding and the head included, as the
    identical ``q8_g32_fp16`` words, and every other tensor as its exact BF16 or FP32 value, the GDN
    gating projections as separate BF16 parents as the official recipes keep them. llama.cpp runs
    the same weights from the GGUF, so a comparison of the two engines compares their arithmetic."""

    if "num_experts" in model.config:
        raise ValueError("this recipe requires Qwen3.5 Dense mathematics")
    for name, parameter in model.parameters.items():
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if not (parameter.projection or name == "text/token_embedding"):
            continue
        try:
            stored = parameter.source.read_encoded(0, 1).format
        except (TypeError, ValueError):  # no encoded rows (None, or a checkpoint's BF16 matrix)
            stored = None
        if stored != Q8:
            raise ValueError(f"{name}: this recipe stores the Q8_0 matrices of a GGUF import (tools.convert --gguf)")
        recipe.assign(name, format=Q8, method=import_encoded)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def qwen3_8_27b_nvfp4_nvidia(model, recipe, sources):
    """nvidia/Qwen3.8-27B-NVFP4 (ModelOpt AutoQuant) layout: every text MLP
    projection is NVFP4; attention and GDN projections are per-row FP8. The
    source output head is NVFP4, but the runtime registers the vocabulary
    projection only for FP8, so it is dequantised and re-quantised there.
    The DFlash2 drafter's MLP gate/up is quantized to NVFP4 (A16), the rest of
    the drafter to Q8."""
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _dflash2_nvfp4_gate_up(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        if name == "text/output_head":
            recipe.assign(
                name,
                format=FP8,
                method=fp8_row_maxabs,
                source=source,
                activation_policy="AllowA8",
            )
            continue
        format = (
            "nvfp4"
            if name.startswith("text/layers/") and "/mlp/" in name
            else FP8
        )
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def qwen3_8_27b_nvfp4_orcarouter(model, recipe, sources):
    """orcarouter/Qwen3.8-27B-Uncensored-NVFP4 (GPTQ compressed-tensors)
    layout: MLP projections of layers 0..55 are NVFP4 packed; the last eight
    layers' MLP projections and every attention/GDN projection are per-row
    FP8; the embedding and output head remain BF16 and are re-quantised."""
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    recipe.assign("text/output_head", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if (
            not name.startswith("text/layers/")
            or not parameter.projection
        ):
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.assign(name, source=model.source(name, quantized))
            continue
        layer = int(name.split("/")[2])
        format = "nvfp4" if ("/mlp/" in name and layer < 56) else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


FP8_F32 = "fp8_e4m3fn_row_f32"
_MODELOPT_FP8_ROLES = (
    "/gdn/query", "/gdn/key", "/gdn/value", "/gdn/z", "/gdn/output",
    "/attention/query", "/attention/gate", "/attention/key", "/attention/value", "/attention/output",
)


def _modelopt_mixed_algorithms(store) -> set[str]:
    """The per-layer algorithms of a ModelOpt ``MIXED_PRECISION`` export's ``quantization_config``."""

    config = store.config.get("quantization_config")
    if not isinstance(config, dict) or config.get("quant_method") != "modelopt":
        raise ValueError("this recipe converts a ModelOpt export (quantization_config.quant_method modelopt)")
    if config.get("quant_algo") != "MIXED_PRECISION" or not isinstance(config.get("quantized_layers"), dict):
        raise ValueError("this recipe converts a ModelOpt MIXED_PRECISION export with quantized_layers")
    return {layer.get("quant_algo") for layer in config["quantized_layers"].values()}


def qwen3_5_moe_nvfp4_modelopt(model, recipe, sources):
    """A ModelOpt ``MIXED_PRECISION`` export of a Qwen3.5 MoE model (Ornith-1.5-122B-NVFP4, design
    docs/maintainer/qwen3_5-122b-offload-design.md §6), bit-exact except for the MTP drafter:

    - Routed experts (weight-only ``W4A16_NVFP4``): exact ``nvfp4_mul`` banks in
      ``nvfp4_expert_rg16_v1`` (codes, E4M3 block scales and the FP32 ``weight_scale_2`` multipliers
      as stored), run with the canonical W4A16 arithmetic; the engine keeps them in host memory
      behind its VRAM expert cache.
    - Shared experts (``W4A16_NVFP4``): one-record banks of the same layout, run as a resident job of
      the routed-expert kernels.
    - ``lm_head`` (``W4A16_NVFP4``): exact ``nvfp4_mul`` in ``nvfp4_rg16_v1``, FP32 logits.
    - Attention and GDN projections (per-tensor ``FP8`` with static ``input_scale``): exact
      ``fp8_e4m3fn_row_f32`` (every row holds its matrix's FP32 ``weight_scale``), ``AllowA8`` with the
      checkpoint's FP32 input scale as the Use's ``activation_input_scale``: static 8-bit
      activations in prefill, 16-bit in decode on the exact route.
    - Every BF16 tensor (router, shared-expert gate, GDN a/b, norms, convolution, ``A_log`` and
      ``dt_bias`` widened exactly to FP32, embedding, vision) keeps its value.
    - The BF16 MTP layer is re-quantized as Qwen3.8-Flash-Next's drafter: ``q8_g32_fp16`` dense,
      ``q4_g64_fp16`` (MSE group scales) routed experts, BF16 router and shared-expert gate. It only
      proposes tokens the model verifies, so its precision changes acceptance, never output.
    - A selected DFlash2 (or DFlash) drafter, ``--source dflash2=PATH`` (Ornith's
      incoai/Ornith-1.5-122B-DFlash2), as the official 27B artifact's (``_dflash_drafters``).
    """

    from .expert_banks import ExpertBankSource, import_expert_bank

    if "num_experts" not in model.config:
        raise ValueError("this recipe requires Qwen3.5 MoE mathematics")
    base = sources["base"]
    unexpected = _modelopt_mixed_algorithms(base) - {"FP8", "W4A16_NVFP4"}
    if unexpected:
        raise ValueError(f"this recipe imports FP8 and W4A16_NVFP4 layers only, not {sorted(unexpected)}")
    banks = [
        name for name in model.parameters
        if name.startswith("text/layers/") and name.endswith(("/moe/experts", "/moe/shared_expert"))
    ]
    if not any(name.endswith("/moe/experts") for name in banks):
        raise ValueError("this recipe imports per-expert ModelOpt NVFP4 routed experts")
    for name in banks:
        source = model.parameters[name].source
        if not isinstance(source, ExpertBankSource) or source.packed or not source.weight_only:
            raise ValueError(f"{name}: expected weight-only ModelOpt NVFP4 experts (no input_scale)")
        recipe.assign(
            name,
            format="nvfp4_mul",
            layout="nvfp4_expert_rg16_v1",
            method=import_expert_bank,
            activation_policy="A16Only",
        )
        recipe.group(name, shape=model.parameters[name].shape)
    for name in model.parameters:
        if name.startswith("text/layers/") and name.endswith(_MODELOPT_FP8_ROLES):
            recipe.assign(
                name,
                format=FP8_F32,
                method=import_encoded,
                source=model.source(name, base, FP8_F32),
                activation_policy="AllowA8",
            )
    recipe.assign(
        "text/output_head",
        format="nvfp4_mul",
        layout="nvfp4_rg16_v1",
        method=import_encoded,
        source=model.source("text/output_head", base, "nvfp4_mul"),
        activation_policy="A16Only",
    )
    _flash_next_drafter(model, recipe)
    _dflash_drafters(model, recipe)


def _flash_next_drafter(model, recipe, experts=Q4):
    """The MTP drafter of both Qwen3.8-Flash-Next recipes. It only proposes tokens that the main
    model verifies, so its precision changes acceptance, never output (design §11.2). Its
    projections are ``q8_g32_fp16`` except the router and shared-expert gate (BF16, as NVIDIA stores
    them), and its 512 routed experts, block-FP8 in NVIDIA's checkpoint, ``q4_g64_fp16`` with
    MSE-chosen group scales (1.34 GB, all VRAM-resident; Strata keeps them at 2.25 bits), or
    ``experts`` (``q2_g64_fp16``: 0.71 GB, the 0.63 GB difference going to expert-cache frames)."""
    for name, parameter in model.parameters.items():
        if not name.startswith("mtp/") or not parameter.projection:
            continue
        if name.endswith(("/moe/router", "/moe/shared_score")):
            continue
        if "/moe/experts/" in name:
            _assign(recipe, name, experts, method=grouped_mse)
        else:
            _assign(recipe, name, Q8)


def qwen3_8_flash_next_nvfp4(model, recipe, sources):
    """Recipe A of Qwen3.8-Flash-Next (design §6.1): the main model and the vision tower bit-exact to
    NVIDIA's checkpoint. Every NVIDIA-quantized tensor keeps its codes and scales (the routed
    experts become exact ``nvfp4_mul`` banks in ``nvfp4_expert_rg16_v1`` with each matrix's own
    ModelOpt input scale; the FP8 n-gram table is written to its own volume) and every BF16 or FP32
    tensor keeps its dtype. Only the MTP drafter is re-quantized (``_flash_next_drafter``)."""
    from .expert_banks import import_expert_bank, weight_only_bank
    from .qwen4_exp import Fp8ExpertBankSource

    if model.config.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("this official recipe requires Qwen4Exp mathematics")
    for name, parameter in model.parameters.items():
        if name.startswith("text/layers/") and name.endswith("/moe/experts"):
            if weight_only_bank(parameter.source):
                raise ValueError(f"{name}: weight-only experts take recipe C (qwen3_8_flash_next_nvfp4_orcarouter)")
            if isinstance(parameter.source, Fp8ExpertBankSource):
                raise ValueError(f"{name}: block-FP8 experts take the FP8 recipe (qwen3_8_flash_next_fp8)")
            recipe.assign(
                name,
                format="nvfp4_mul",
                layout="nvfp4_expert_rg16_v1",
                method=import_expert_bank,
                activation_policy="AllowA4",
            )
            recipe.group(name, shape=parameter.shape)
    _flash_next_drafter(model, recipe)


def qwen3_8_flash_next_nvfp4_dense8(model, recipe, sources):
    """Recipe B of Qwen3.8-Flash-Next (design §6.1): recipe A, plus the dense projection classes
    that dominate per-token weight reads in ``q8_g32_fp16`` (W8A16): the GDN q/k/v/z and output
    projections, the QSA output projection, the shared experts, the hyper-connection mixers, the
    PLE projections and ``lm_head``. ``q8_g32_fp16`` is used for every class rather than FP8 rows: a
    32-element group scale and an 8-bit integer give far finer resolution at 1.06 bytes per weight.
    The router, shared-expert gate, GDN a/b, the QSA query/gate/key/value/indexer group and the
    embedding stay as recipe A has them, and so does the drafter."""

    qwen3_8_flash_next_nvfp4(model, recipe, sources)
    _flash_next_dense8(model, recipe)


def _flash_next_dense8(model, recipe):
    """Recipe B's dense projection classes in ``q8_g32_fp16`` (W8A16)."""

    dense8 = (
        "/attn_hc/down", "/attn_hc/inject", "/attn_hc/up",
        "/mlp_hc/down", "/mlp_hc/inject", "/mlp_hc/up",
        "/gdn/query", "/gdn/key", "/gdn/value", "/gdn/z", "/gdn/output",
        "/attention/output",
        "/moe/shared/gate", "/moe/shared/up", "/moe/shared/down",
        "/ple/key_projection", "/ple/value_projection",
    )
    for name in model.parameters:
        if name.startswith("text/layers/") and name.endswith(dense8):
            _assign(recipe, name, Q8)
    for name in ("text/final_mixer/down", "text/final_mixer/up", "text/output_head"):
        _assign(recipe, name, Q8)


def qwen3_8_flash_next_fp8(model, recipe, sources):
    """Qwen3.8-Flash-Next's official FP8 checkpoint (docs/maintainer/qwen3_8-flash-next-design.md §19.3.25): Qwen's official
    FP8 checkpoint bit-exact. The routed experts keep their E4M3FN codes and BF16 128 x 128 tile scales
    (``fp8_e4m3fn_b128`` in ``fp8_expert_b128_v1``, run W8A16 with exact arithmetic); every BF16 or
    FP32 tensor keeps its dtype; the n-gram table is the FP8 volume recipes A-C write. Only the MTP
    drafter is re-quantized (``_flash_next_drafter``), its routed experts in ``q2_g64_fp16``: the
    0.63 GB saved against ``q4_g64_fp16`` goes to expert-cache frames."""

    from .qwen4_exp import Fp8ExpertBankSource, import_fp8_expert_bank

    if model.config.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("this official recipe requires Qwen4Exp mathematics")
    banks = [name for name in model.parameters if name.startswith("text/layers/") and name.endswith("/moe/experts")]
    if not banks or not all(isinstance(model.parameters[name].source, Fp8ExpertBankSource) for name in banks):
        raise ValueError("the FP8 recipe imports Qwen's block-FP8 experts (Qwen/Qwen3.8-Flash-Next-FP8)")
    for name in banks:
        recipe.assign(name, format="fp8_e4m3fn_b128", layout="fp8_expert_b128_v1", method=import_fp8_expert_bank,
                      activation_policy="A16Only")
        recipe.group(name, shape=model.parameters[name].shape)
    _flash_next_drafter(model, recipe, experts=Q2)


def qwen3_8_flash_next_nvfp4_orcarouter(model, recipe, sources):
    """Recipe C of Qwen3.8-Flash-Next (design §6.1): orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4,
    a weight-only compressed-tensors export of the abliterated model, served with BF16 activations.

    Bit-exact where the checkpoint is quantized: the routed experts' NVFP4 codes and block scales (a
    W4A16 bank, design §16.2.1: the global scale stored as its FP32 reciprocal, no activation scales)
    and the row-scaled FP8 GDN, QSA and shared-expert projections (W8A16). Its BF16 classes follow
    recipe B: the hyper-connection mixers, the PLE projections and ``lm_head`` in ``q8_g32_fp16``;
    the router, shared-expert gate, GDN a/b, QSA indexer, embedding and vision tower stay BF16. The
    n-gram table is quantized per tensor (qwen4_exp.NgramTable), which reproduces NVIDIA's FP8 table
    of the same base model word for word. The drafter is recipe A's, from the checkpoint's BF16 MTP.

    The QSA query/gate/key/value projections (FP8) and the indexer (BF16) form two parents, which the
    model runs as two projections."""

    from .expert_banks import import_expert_bank, weight_only_bank

    if model.config.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("this official recipe requires Qwen4Exp mathematics")
    banks = [name for name in model.parameters if name.startswith("text/layers/") and name.endswith("/moe/experts")]
    if not banks or not all(weight_only_bank(model.parameters[name].source) for name in banks):
        raise ValueError("recipe C imports weight-only compressed-tensors experts (W4A16 banks)")
    base = sources["base"]
    fp8 = (
        "/gdn/query", "/gdn/key", "/gdn/value", "/gdn/z", "/gdn/output",
        "/attention/query", "/attention/gate", "/attention/key", "/attention/value", "/attention/output",
        "/moe/shared/gate", "/moe/shared/up", "/moe/shared/down",
    )
    dense8 = (
        "/attn_hc/down", "/attn_hc/inject", "/attn_hc/up",
        "/mlp_hc/down", "/mlp_hc/inject", "/mlp_hc/up",
        "/ple/key_projection", "/ple/value_projection",
    )
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/"):
            continue
        if name.endswith("/moe/experts"):
            recipe.assign(
                name,
                format="nvfp4_mul",
                layout="nvfp4_expert_rg16_v1",
                method=import_expert_bank,
                activation_policy="A16Only",
            )
            recipe.group(name, shape=parameter.shape)
        elif name.endswith(fp8):
            recipe.assign(
                name,
                format=FP8,
                method=import_encoded,
                source=model.source(name, base, FP8),
                activation_policy="A16Only",
            )
        elif name.endswith(dense8):
            _assign(recipe, name, Q8)
    for name in ("text/final_mixer/down", "text/final_mixer/up", "text/output_head"):
        _assign(recipe, name, Q8)
    for layer, kind in enumerate(model.config["layer_types"]):
        if kind == "full_attention":
            p = f"text/layers/{layer}/attention/"
            recipe.group([p + "query", p + "gate", p + "key", p + "value"])
            recipe.group([p + "index_query", p + "index_key"])
    _flash_next_drafter(model, recipe)


MXFP8 = "fp8_e4m3fn_g32_e8m0"


def qwen3_8_flash_next_fp4_local(model, recipe, sources):
    """nvidia/Qwen3.8-Flash-Next-FP4-Local (design §6.1.2): NVIDIA's quantization-aware-distilled
    checkpoint, bit-exact, run by default in the arithmetic NVIDIA serves it with: 16-bit activations
    everywhere.

    - The routed experts keep their NVFP4 codes, their block scales (stored in the checkpoint in the
      lossless NVFP4-CSF encoding, restored exactly) and their FP32 multipliers as ``nvfp4_mul``
      banks in ``nvfp4_expert_rg16_v1``, with their calibrated FP32 activation scales. Their Use
      permits 4-bit activations, and the text config's ``expert_a4_activations: optional`` makes the
      16-bit arithmetic (design §16.2.1) the default: the engine runs 4-bit expert activations
      (§16.2) only with ``--expert-a4-activations``.
    - The MXFP8 GDN q/k/v/z and output, QSA query/gate/key/value, indexer and output and
      shared-expert projections keep their E4M3FN codes and UE8M0 group scales
      (``fp8_e4m3fn_g32_e8m0``, W8A16).
    - The GDN a/b projections (MXFP8, a parameter of their own) and the vision attention become
      their exact BF16 values.
    - The vision MLPs and merger (weight-only NVFP4) keep their words as dense ``nvfp4_mul``
      matrices in ``nvfp4_rg16_v1`` (``_fp4_local_vision``), run with the canonical W4A16
      arithmetic, so no main-model or vision weight is rounded.
    - Every BF16 or FP32 tensor keeps its dtype; the NVFP4 n-gram table is written verbatim to its
      own volume with its FP32 multiplier.
    - The MTP drafter is re-quantized as recipes A-C build it (``_flash_next_drafter``) from its
      weight-only NVFP4 experts' exact values and its BF16 dense projections.

    The QSA query/gate/key/value and indexer projections form one MXFP8 parent [13952, 2560], which
    the model runs as one projection (the builder's grouping, as recipe A's).

    The published conversion also passes ``--proposal --screen-heads``: the proposal head, then the
    2-bit screens of the LM head and the proposal head (tools/convert/screen.py, 273 MB), extra objects
    after every other, so the tensors this recipe assigns keep their bytes."""

    from .expert_banks import ExpertBankSource, import_expert_bank, weight_only_bank

    if model.config.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("this official recipe requires Qwen4Exp mathematics")
    banks = [name for name in model.parameters if name.startswith("text/layers/") and name.endswith("/moe/experts")]
    if not banks or not all(
        isinstance(model.parameters[name].source, ExpertBankSource) and not weight_only_bank(model.parameters[name].source)
        for name in banks
    ):
        raise ValueError("the FP4-Local recipe imports ModelOpt NVFP4 experts (nvidia/Qwen3.8-Flash-Next-FP4-Local)")
    if model.config["ngram_table"]["format"] != "nvfp4":
        raise ValueError("the FP4-Local recipe binds the checkpoint's own NVFP4 n-gram table")
    base = sources["base"]
    mxfp8 = (
        "/gdn/query", "/gdn/key", "/gdn/value", "/gdn/z", "/gdn/output",
        "/attention/query", "/attention/gate", "/attention/key", "/attention/value",
        "/attention/index_query", "/attention/index_key", "/attention/output",
        "/moe/shared/gate", "/moe/shared/up", "/moe/shared/down",
    )
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/"):
            continue
        if name.endswith("/moe/experts"):
            recipe.assign(
                name,
                format="nvfp4_mul",
                layout="nvfp4_expert_rg16_v1",
                method=import_expert_bank,
                activation_policy="AllowA4",
            )
            recipe.group(name, shape=parameter.shape)
        elif name.endswith(mxfp8):
            recipe.assign(
                name,
                format=MXFP8,
                method=import_encoded,
                source=model.source(name, base, MXFP8),
                activation_policy="A16Only",
            )
    # The banks keep their activation scales and permit 4-bit activations; 16-bit is the default.
    model.config["expert_a4_activations"] = "optional"
    _fp4_local_vision(model, recipe, base)
    _flash_next_drafter(model, recipe)


# FP4-Local's weight-only NVFP4 vision matrices: each layer's MLP and the merger.
_FP4_LOCAL_VISION_NVFP4 = ("/mlp/fc1", "/mlp/fc2", "/merger/fc1", "/merger/fc2")


def _fp4_local_vision(model, recipe, base):
    """FP4-Local's vision MLPs ([4304, 1152], [1152, 4304]) and merger ([4608, 4608], [2560, 4608]):
    NVIDIA's E2M1 codes, E4M3 block scales and FP32 ``weight_scale_2`` words as dense ``nvfp4_mul``
    matrices in ``nvfp4_rg16_v1``, run with 16-bit activations by the canonical W4A16 projection
    (rg16_linear), as the Qwen3.5-122B head. Each matrix is its own parent, so the artifact's other
    objects keep their order and bytes. Without the vision component this assigns nothing."""

    for name in model.parameters:
        if name.startswith("vision/") and name.endswith(_FP4_LOCAL_VISION_NVFP4):
            recipe.assign(
                name,
                format="nvfp4_mul",
                layout="nvfp4_rg16_v1",
                method=import_encoded,
                source=model.source(name, base, "nvfp4_mul"),
                activation_policy="A16Only",
            )


def qwen3_8_flash_next_gsq_rco_iq3s(model, recipe, sources):
    """ISTA-DASLab's GSQ-RCO IQ3_S GGUF of Qwen3.8-Flash-Next, bit-exact
    (docs/maintainer/qwen3_8-flash-next-design.md §19.3.25). Every GGUF block format is stored verbatim:
    dense matrices of the tiled formats (Q4_K, Q5_K, Q6_K, IQ4_XS) in ``ggml_tiles16_v1`` (the same
    blocks in 16-row tiles, so a tile's block is one contiguous run for the narrow kernels), the
    others and the token embedding (a row gather) in ``ggml_rows_v1``, each in its own per-tensor
    ggml format; the routed experts as ``ggml_expert_*_v1`` banks (gate/up format, the layout naming
    the down format), the one Q8_0 tensor as the identical ``q8_g32_fp16`` words, BF16 and FP32
    tensors in their dtype, and the zero-centred norms as the BF16 values llama.cpp shifted by one
    (qwen4_exp_gguf checks every one is exact). The arithmetic is A16 everywhere. Only the MTP
    drafter, which the GGUF lacks, is built from Qwen's checkpoint and re-quantized as recipes A-C
    build it (``_flash_next_drafter``), its routed experts in ``q2_g64_fp16`` (Strata's 2.25-bit
    drafter precision): the drafter only proposes tokens, and against ``q4_g64_fp16`` experts MTP 4
    + n-gram 7 decoded 201.4-203.1 against 196.1-196.7 tok/s (2026-10-16), with 0.63 GB of VRAM more
    for expert-cache frames.
    """

    from tools.artifact.layouts import GGML_TILES16_FORMATS

    from .qwen4_exp_gguf import GgmlExpertBank, import_ggml_expert_bank
    from .methods import import_ggml_rows

    if model.config.get("architectures") != ["Qwen4ExpForCausalLM"] or model.config.get("gdn_a") != "direct":
        raise ValueError("this recipe requires a Qwen4Exp model built from a GGUF (tools.convert --gguf)")
    for name, parameter in model.parameters.items():
        if not name.startswith("text/"):
            continue
        source = parameter.source
        if isinstance(source, GgmlExpertBank):
            recipe.assign(name, format=source.gate_up_format, layout=source.layout, method=import_ggml_expert_bank,
                          activation_policy="A16Only")
            recipe.group(name, shape=parameter.shape)
        elif source.read_encoded is not None:
            stored = source.read_encoded(0, 1).format
            if stored == Q8:
                recipe.assign(name, format=stored, method=import_encoded)
            elif stored in GGML_TILES16_FORMATS and name != "text/token_embedding":
                recipe.assign(name, format=stored, layout="ggml_tiles16_v1", method=import_ggml_rows)
            else:
                recipe.assign(name, format=stored, method=import_ggml_rows)
    _flash_next_drafter(model, recipe, experts=Q2)


def qwen3_8_flash_next_dflash2(model, recipe, sources):
    """A Qwen3.8-Flash-Next DFlash2 drafter as a companion artifact (``tools.convert --components
    dflash2``), quantized as NInfer's official Qwen3.8-27B artifact stores its DFlash2: every
    projection in ``q8_g32_fp16`` (absmax group scales; the context K/V roles share the query
    block's K/V words) except the dynamic-convolution kernel projections and the selector's hidden
    projection, which stay BF16 with the norms, base kernels and codebooks. The drafter only
    proposes tokens the main model verifies, so its precision changes acceptance, never output."""

    if set(model.components) != {"dflash2"} or model.config.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("this recipe converts a Qwen4Exp DFlash2 companion (--components dflash2)")
    _optional(model, recipe)


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_q6": qwen3_8_27b_q6,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_8_27b_nvfp4_nvidia": qwen3_8_27b_nvfp4_nvidia,
    "qwen3_8_27b_nvfp4_orcarouter": qwen3_8_27b_nvfp4_orcarouter,
    "qwen3_5_gguf_q8_0": qwen3_5_gguf_q8_0,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen3_5_moe_nvfp4_modelopt": qwen3_5_moe_nvfp4_modelopt,
    "qwen3_8_flash_next_nvfp4": qwen3_8_flash_next_nvfp4,
    "qwen3_8_flash_next_nvfp4_dense8": qwen3_8_flash_next_nvfp4_dense8,
    "qwen3_8_flash_next_nvfp4_orcarouter": qwen3_8_flash_next_nvfp4_orcarouter,
    "qwen3_8_flash_next_fp8": qwen3_8_flash_next_fp8,
    "qwen3_8_flash_next_fp4_local": qwen3_8_flash_next_fp4_local,
    "qwen3_8_flash_next_gsq_rco_iq3s": qwen3_8_flash_next_gsq_rco_iq3s,
    "qwen3_8_flash_next_dflash2": qwen3_8_flash_next_dflash2,
}
