"""PLE n-gram row ids: the loop specification against an independent tensor formulation."""

from __future__ import annotations

import torch

from tools.flash_next.ngram import FIXTURE_CONFIG, NgramConfig, fixture_history, head_tables, layer_multipliers, row_ids


def _tensor_rows(c: NgramConfig, layer: int, history: list[int]) -> torch.Tensor:
    """Whole-sequence form: shift right within EOS-delimited segments, then XOR-mix and reduce."""

    t = torch.tensor([history], dtype=torch.long)
    idx = torch.arange(t.shape[1])
    eos_pos = torch.where(t == c.eos_token_id, idx, -1)
    prev = torch.cat([eos_pos.new_full((1, 1), -1), torch.cummax(eos_pos, 1).values[:, :-1]], 1)
    pos_in_segment = idx - (prev + 1)
    shifted = [t]
    for n in range(1, c.ngram_size):
        src = idx - n
        g = t.gather(1, src.clamp(min=0).unsqueeze(0))
        shifted.append(torch.where((pos_in_segment >= n) & (src >= 0), g, torch.full_like(g, c.eos_token_id)))
    mult = torch.tensor(layer_multipliers(c, layer), dtype=torch.long)
    sizes, offsets, _ = head_tables(c, layer)
    sizes, offsets = torch.tensor(sizes), torch.tensor(offsets)
    blocks = []
    for n in range(2, c.ngram_size + 1):
        mix = shifted[0] * mult[0]
        for k in range(1, n):
            mix = torch.bitwise_xor(mix, shifted[k] * mult[k])
        h = slice((n - 2) * c.heads_per_ngram, (n - 1) * c.heads_per_ngram)
        blocks.append(torch.remainder(mix[0].unsqueeze(-1), sizes[h]) + offsets[h])
    return torch.cat(blocks, -1)


def test_rows_match_the_tensor_formulation():
    c = FIXTURE_CONFIG
    history = fixture_history()
    for layer in (0, 1, 3):
        ref = _tensor_rows(c, layer, history)[c.ngram_size - 1 :]
        got = torch.tensor(row_ids(c, layer, history, len(history) - (c.ngram_size - 1)))
        assert torch.equal(got, ref)


def test_mixing_never_leaves_int64_and_rows_stay_in_the_table():
    c = NgramConfig(vocab_size=248320, eos_token_id=248046, ngram_size=4, heads_per_ngram=2,
                    ngram_vocab_size_base=500000, make_ngram_vocab_size_divisible_by=64)
    for layer in range(3):
        mult = layer_multipliers(c, layer)
        assert all(m % 2 == 1 and m * (c.vocab_size - 1) < 2**63 for m in mult)
        _, _, padded = head_tables(c, layer)
        hist = [c.eos_token_id] * 3 + [c.vocab_size - 1 - i for i in range(50)]
        assert all(0 <= r < padded for row in row_ids(c, layer, hist, 50) for r in row)


def test_head_primes_continue_across_layers():
    c = FIXTURE_CONFIG
    s0, o0, _ = head_tables(c, 0)
    s1, _, _ = head_tables(c, 1)
    assert s0[0] == 1000003 and s0 == sorted(set(s0)) and s1[0] > s0[-1]
    assert o0 == [sum(s0[:i]) for i in range(len(s0))]
    try:
        import sympy
    except ImportError:
        return
    p = c.ngram_vocab_size_base - 1
    for size in s0 + s1:
        p = sympy.nextprime(p)
        assert size == p


def test_eos_closes_every_window():
    c = FIXTURE_CONFIG
    e = c.eos_token_id
    # Position after an EOS hashes like the sequence start.
    a = row_ids(c, 0, [e, e, 5], 1)[0]
    b = row_ids(c, 0, [7, e, 5], 1)[0]
    assert a == b
