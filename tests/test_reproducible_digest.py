"""tools/reproducible_digest.py: request files, request-log digests and their comparison (no GPU)."""

import json

import pytest

from tools import reproducible_digest as rd


def write_log(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def done(label, ids, request_id=1):
    return {"event": "request_done", "server_instance_id": "s",
            "request": {"request_id": request_id, "label": label},
            "result": {"finish_reason": "length", "generated_token_ids": ids}}


def test_digests_follow_the_ids_and_ignore_other_requests(tmp_path):
    log = tmp_path / "a.jsonl"
    write_log(log, [{"event": "server_start", "server_instance_id": "s"},
                    done("warm-1", [9, 9, 9], 1),
                    {"event": "request_start", "request": {"request_id": 2, "label": "greedy"}},
                    done("greedy", [1, 2, 3], 2),
                    done("seeded", [4, 5], 3),
                    done(None, [7], 4)])
    digests = rd.digests_from_log(log, ["greedy", "seeded"])
    assert set(digests) == {"greedy", "seeded"}
    assert digests["greedy"][1] == 3 and digests["seeded"][1] == 2
    other = tmp_path / "b.jsonl"
    write_log(other, [done("seeded", [4, 5], 8), done("greedy", [1, 2, 4], 9)])
    assert rd.compare(digests, rd.digests_from_log(other, ["greedy", "seeded"]), ["greedy", "seeded"]) == ["greedy"]


def test_a_missing_or_repeated_request_is_an_error(tmp_path):
    log = tmp_path / "a.jsonl"
    write_log(log, [done("greedy", [1])])
    with pytest.raises(ValueError, match="no request_done"):
        rd.digests_from_log(log, ["greedy", "seeded"])
    write_log(log, [done("greedy", [1], 1), done("greedy", [1], 2)])
    with pytest.raises(ValueError, match="twice"):
        rd.digests_from_log(log, ["greedy"])
    write_log(log, [{"event": "request_done", "request": {"label": "greedy"}, "result": {}}])
    with pytest.raises(ValueError, match="generated_token_ids"):
        rd.digests_from_log(log, ["greedy"])


def test_request_files_need_unique_labels_and_bodies(tmp_path):
    path = tmp_path / "requests.json"
    path.write_text(json.dumps([{"label": "x", "body": {"messages": []}}]), encoding="utf-8")
    assert rd.read_requests(path)[0]["label"] == "x"
    for bad in ([], [{"label": "x"}], [{"label": "x", "body": {}}, {"label": "x", "body": {}}]):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            rd.read_requests(path)
