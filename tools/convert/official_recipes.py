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

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None, method=grouped_absmax):
    method = method if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


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
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8, method=method)
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


def _flash_next_drafter(model, recipe):
    """The MTP drafter of both Qwen3.8-Flash-Next recipes. It only proposes tokens that the main
    model verifies, so its precision changes acceptance, never output (design §11.2). Its
    projections are ``q8_g32_fp16`` except the router and shared-expert gate (BF16, as NVIDIA stores
    them), and its 512 routed experts, block-FP8 in NVIDIA's checkpoint, ``q4_g64_fp16`` with
    MSE-chosen group scales (1.34 GB, all VRAM-resident; Strata keeps them at 2.25 bits)."""
    for name, parameter in model.parameters.items():
        if not name.startswith("mtp/") or not parameter.projection:
            continue
        if name.endswith(("/moe/router", "/moe/shared_score")):
            continue
        if "/moe/experts/" in name:
            _assign(recipe, name, Q4, method=grouped_mse)
        else:
            _assign(recipe, name, Q8)


def qwen3_8_flash_next_nvfp4(model, recipe, sources):
    """Recipe A of Qwen3.8-Flash-Next (design §6.1): the main model and the vision tower bit-exact to
    NVIDIA's checkpoint. Every NVIDIA-quantized tensor keeps its codes and scales (the routed
    experts become exact ``nvfp4_mul`` banks in ``nvfp4_expert_rg16_v1`` with each matrix's own
    ModelOpt input scale; the FP8 n-gram table is written to its own volume) and every BF16 or FP32
    tensor keeps its dtype. Only the MTP drafter is re-quantized (``_flash_next_drafter``)."""
    from .qwen4_exp import import_expert_bank

    if model.config.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("this official recipe requires Qwen4Exp mathematics")
    for name, parameter in model.parameters.items():
        if name.startswith("text/layers/") and name.endswith("/moe/experts"):
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


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_q6": qwen3_8_27b_q6,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_8_27b_nvfp4_nvidia": qwen3_8_27b_nvfp4_nvidia,
    "qwen3_8_27b_nvfp4_orcarouter": qwen3_8_27b_nvfp4_orcarouter,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen3_8_flash_next_nvfp4": qwen3_8_flash_next_nvfp4,
    "qwen3_8_flash_next_nvfp4_dense8": qwen3_8_flash_next_nvfp4_dense8,
}
