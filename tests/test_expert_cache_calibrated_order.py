"""The calibrated expert order generator (tools/expert_cache_replay/calibrated_order.py, candidates §23 NU-43) on
synthetic route traces: requests split per lane and normalized per request, only rounds of at most --max-columns
columns counted, the arms mixed by weight, ties to the lower expert id; the C++ table it writes reads back, its
saved state follows expert_cache::calibrated_order_state, and the committed 122B table is empty or a valid one. The
collection driver's task set loads from the repository's fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from tools.expert_cache_replay import calibrated_order as co
from tools.expert_cache_replay import collect_traces as ct
from tools.expert_cache_replay import host_tier as ht

LAYERS, EXPERTS, TOP_K = 2, 6, 2
HEADER = {'layers': LAYERS, 'experts': EXPERTS, 'top_k': TOP_K, 'frames': 0, 'lanes': 1, 'max_columns': 128,
          'max_width': 1, 'prefill_chunk': 128}


def _round(kind, width, experts, position=ht.UNKNOWN_POSITION, lane=0):
    """One row of `width` columns, every column routing `experts` (TOP_K ids) in every layer."""
    routes = np.tile(np.asarray(experts, np.int32), (LAYERS, width, 1))
    return ht.Round(kind, 1, width, width, 0, position, np.array([lane], np.int32), np.ones(width, bool), routes)


def _request(prompt, prompt_experts, decode, decode_experts, lane=0):
    rounds = [_round(ht.PREFILL, prompt, prompt_experts, position=0, lane=lane)]
    rounds += [_round(ht.DECODE, 1, decode_experts, lane=lane) for _ in range(decode)]
    return rounds


def _trace(tmp_path, name, rounds):
    path = tmp_path / name
    ht.write_trace(path, HEADER, rounds)
    return path


def test_requests_weigh_equally_and_wide_rounds_are_not_counted(tmp_path):
    # A long request decodes on experts 4, 5; a short one on 1, 2. Their 100-column prompts route 0, 3, which the
    # 64-column limit leaves out. Normalized per request, the four decode experts tie and rank by id.
    path = _trace(tmp_path, 'plain.bin', _request(100, [0, 3], 200, [4, 5]) + _request(100, [0, 3], 2, [1, 2]))
    counts, rounds, columns = co.request_counts(path, 64)
    assert len(counts) == 2 and rounds == 202 and columns == 202
    order, score, shape, counted, used = co.calibrated_order({'plain': [path], 'mtp': [], 'dflash': []},
                                                             dict(zip(co.ARMS, co.DEFAULT_WEIGHTS)))
    assert shape == (LAYERS, EXPERTS, TOP_K) and used == {'plain': 1.0} and counted['plain'].requests == 2
    assert order.tolist() == [[1, 2, 4, 5, 0, 3]] * LAYERS
    np.testing.assert_allclose(score[0], [0, 0.25, 0.25, 0, 0.25, 0.25])
    # Without the limit the prompts count too: 0 and 3 lead (each request routes them in 100 of its columns).
    wide, _, _, _, _ = co.calibrated_order({'plain': [path], 'mtp': [], 'dflash': []},
                                           dict(zip(co.ARMS, co.DEFAULT_WEIGHTS)), max_columns=1000)
    assert wide[0][:2].tolist() == [0, 3]


def test_a_prefill_after_decoding_starts_the_next_request_on_its_lane(tmp_path):
    # Two requests whose second prompt does not start at position 0 (a reused prefix): still two requests.
    rounds = _request(8, [0, 1], 3, [2, 3])
    rounds += [_round(ht.PREFILL, 8, [0, 1], position=64)] + [_round(ht.DECODE, 1, [4, 5])]
    counts, _, _ = co.request_counts(_trace(tmp_path, 't.bin', rounds), 64)
    assert len(counts) == 2
    assert counts[1][0].tolist() == [8, 8, 0, 0, 1, 1]


def test_arms_mix_by_weight_renormalized_over_the_arms_given(tmp_path):
    plain = _trace(tmp_path, 'plain.bin', _request(4, [0, 1], 4, [0, 1]))
    mtp = _trace(tmp_path, 'mtp.bin', _request(4, [2, 3], 4, [2, 3]))
    order, score, _, _, used = co.calibrated_order({'plain': [plain], 'mtp': [mtp], 'dflash': []},
                                                   dict(zip(co.ARMS, co.DEFAULT_WEIGHTS)))
    assert used == pytest.approx({'plain': 2 / 3, 'mtp': 1 / 3})
    assert order[0].tolist() == [0, 1, 2, 3, 4, 5]
    swapped, _, _, _, _ = co.calibrated_order({'plain': [plain], 'mtp': [mtp], 'dflash': []},
                                              {'plain': 0.25, 'mtp': 0.5, 'dflash': 0.25})
    assert swapped[0].tolist() == [2, 3, 0, 1, 4, 5]


def test_traces_of_another_shape_are_refused(tmp_path):
    plain = _trace(tmp_path, 'plain.bin', _request(4, [0, 1], 4, [0, 1]))
    other = tmp_path / 'other.bin'
    ht.write_trace(other, dict(HEADER, experts=8), _request(4, [0, 1], 4, [0, 1]))
    with pytest.raises(ValueError, match='other traces'):
        co.calibrated_order({'plain': [plain], 'mtp': [other], 'dflash': []}, dict(zip(co.ARMS, co.DEFAULT_WEIGHTS)))
    with pytest.raises(ValueError, match='other traces'):
        co.calibrated_order({'plain': [plain], 'mtp': [], 'dflash': []}, dict(zip(co.ARMS, co.DEFAULT_WEIGHTS)),
                            expect=(48, 256, 8))


def test_the_table_reads_back_and_its_state_is_the_engines(tmp_path):
    plain = _trace(tmp_path, 'plain.bin', _request(4, [0, 1], 6, [5, 2]) + _request(4, [0, 1], 6, [5, 3]))
    order, _, shape, counted, used = co.calibrated_order({'plain': [plain], 'mtp': [], 'dflash': []},
                                                         dict(zip(co.ARMS, co.DEFAULT_WEIGHTS)))
    table = tmp_path / 'calibrated_order.inc'
    table.write_text(co.format_table(order, shape, counted, used, 64, generated='test'), encoding='utf-8')
    text = table.read_text(encoding='utf-8')
    assert 'constexpr std::uint32_t kTableLayers  = 2;' in text and 'kTable{{' in text and text.rstrip().endswith('}};')
    read_shape, read_order = co.read_table(table)
    assert read_shape == shape and (read_order == order).all()
    # expert_cache::calibrated_order_state: counts are the ranks reversed, the ranking interleaves the layers (the
    # C++ test's small order: layer 0 ranks 2, 0, 3, 1; layer 1 ranks 1, 3, 0, 2).
    counts, ranked = co.order_state(np.array([[2, 0, 3, 1], [1, 3, 0, 2]]))
    assert counts.tolist() == [3, 1, 4, 2, 2, 4, 1, 3]
    assert ranked.tolist() == [2, 5, 0, 7, 3, 4, 1, 6]


def test_the_committed_122b_table_is_empty_or_valid():
    table = co.read_table()
    if table is not None:
        shape, order = table
        assert shape == (48, 256, 8) and order.shape == (48, 256)


def test_a_table_whose_row_repeats_an_expert_is_refused(tmp_path):
    path = tmp_path / 'bad.inc'
    path.write_text('constexpr std::uint32_t kTableLayers  = 1;\nconstexpr std::uint32_t kTableExperts = 3;\n'
                    'constexpr std::uint32_t kTableTopK    = 1;\nconstexpr std::array<std::uint16_t, 3> kTable{{\n'
                    '    0, 0, 2,\n}};\n', encoding='utf-8')
    with pytest.raises(ValueError, match='permutation'):
        co.read_table(path)


def test_the_collection_task_set_loads_from_the_repository():
    tasks = ct.fixture_tasks(512)
    domains = {domain for _, domain, _ in tasks}
    assert domains == set(ct.FIXTURES) and len(tasks) == sum(len(v) for v in ct.FIXTURES.values())
    assert all(body['max_completion_tokens'] <= 512 and body['messages'] for _, _, body in tasks)
