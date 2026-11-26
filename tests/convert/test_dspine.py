"""The DSpine drafter's conversion (tools/convert/dspine.py): its stored config and its derived sender table."""

import torch

from tools.convert import dspine


def _raw():
    return {
        "architectures": ["DSpineDraftModel"], "model_type": "qwen3", "hidden_size": 5120, "intermediate_size": 17408,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 5,
        "max_position_embeddings": 262144, "vocab_size": 248320, "num_target_layers": 64, "block_size": 16,
        "rms_norm_eps": 1e-6, "layer_types": ["sliding_attention"] * 5, "sliding_window": 8192,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}, "rope_theta": 10000000,
        "code_bridge_second_pass": {"first_exact": 4, "head_topk": 16, "mode": "serial_table"},
        "dflash_config": {
            "mask_token_id": 248070, "target_layer_ids": [1, 16, 31, 46, 61],
            "selected_prefix_code_bridge": {"code_size": 1024, "eps": 1e-6, "kappa": 10.0, "rank": 512},
        },
    }


def test_component_config_is_the_engines_dspine_config():
    config = dspine.component_config(_raw())
    assert dspine.is_dspine(config)
    assert config["rope_parameters"] == {"rope_theta": 1e7}
    assert config["dspine_config"] == {
        "target_layer_ids": [1, 16, 31, 46, 61], "mask_token_id": 248070, "code_size": 1024, "rank": 512,
        "bridge_eps": 1e-6, "head_top_k": 16}
    assert "dflash_config" not in config and "code_bridge_second_pass" not in config


def test_sender_table_is_the_rounded_fp64_product_of_bf16_codes():
    generator = torch.Generator().manual_seed(3)
    codebook = torch.randn(37, 64, generator=generator).to(torch.bfloat16)
    predecessor = torch.randn(16, 64, generator=generator).to(torch.bfloat16)
    value = torch.randn(16, 64, generator=generator).to(torch.bfloat16)
    table = dspine.sender_table(codebook, predecessor, value, 1e-6, rows_per_chunk=10)
    assert table.dtype == torch.bfloat16 and tuple(table.shape) == (37, 32)
    x = codebook.float()
    codes = ((64 ** 0.5) * x / (x.norm(dim=1, keepdim=True) + 1e-6)).to(torch.bfloat16).double()
    expected = (codes @ torch.cat([predecessor, value], 0).double().T).to(torch.bfloat16)
    assert torch.equal(table.view(torch.int16), expected.view(torch.int16))
    # Each row is its own code's: the chunking changes nothing.
    assert torch.equal(dspine.sender_table(codebook, predecessor, value, 1e-6).view(torch.int16),
                       table.view(torch.int16))
