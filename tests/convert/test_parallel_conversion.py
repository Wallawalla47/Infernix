import json
import threading
import time

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.reader import Artifact
from tools.artifact.writer import ArtifactWriter
from tools.convert import pipeline
from tools.convert.model import Model, Parameter
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source
from tools.convert.sources.safetensors import SafetensorsSource, tensor_source


def _direct_model():
    values = torch.arange(128, dtype=torch.float32).to(torch.bfloat16).reshape(1, 128)
    model = Model({"text": {"config": {}}})
    model.add(Parameter("weight", tuple(values.shape), array_source(values, "weight")))
    return model, Recipe(model)


def _object_bytes(path):
    with Artifact(path) as artifact:
        return {
            obj.id: b"".join(artifact.iter_range(obj.offset, obj.bytes))
            for obj in artifact.objects
        }


_SHAPE = (128, 128)


def _sharded_source(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    shape = _SHAPE
    weight_map = {}

    # Use more shards than SafetensorsSource's four-FD cache so parallel reads
    # exercise descriptor eviction as well as concurrent conversion.
    for index in range(8):
        name = f"weight_{index}"
        shard = f"model-{index + 1:05d}-of-00008.safetensors"
        values = (
            (
                torch.arange(shape[0] * shape[1], dtype=torch.float32)
                + index * 17
            ).remainder(257)
            - 128
        ).to(torch.bfloat16).reshape(shape)

        save_file({name: values}, source_dir / shard)
        weight_map[name] = shard

    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    (source_dir / "config.json").write_text("{}", encoding="utf-8")
    return source_dir


def _convert_sharded(source_dir, output, **options):
    with SafetensorsSource(source_dir) as reader:
        model = Model({"text": {"config": {}}})

        for index in range(8):
            name = f"weight_{index}"
            model.add(
                Parameter(
                    name,
                    _SHAPE,
                    tensor_source(reader, name, _SHAPE),
                    inputs=("input",),
                )
            )

        recipe = Recipe(model)
        recipe.assign(
            "weight_*",
            format="q8_g32_fp16",
            method="grouped_absmax",
        )

        return convert(
            model,
            recipe,
            output,
            device="cpu",
            rows_per_chunk=32,
            **options,
        )


def test_parallel_cpu_conversion_matches_serial_with_sharded_sources(tmp_path):
    source_dir = _sharded_source(tmp_path)
    outputs = {}

    for workers in (1, 4):
        output = tmp_path / f"workers-{workers}.infernix"
        report = _convert_sharded(source_dir, output, workers=workers)
        assert report["workers"] == workers
        outputs[workers] = _object_bytes(output)

    assert outputs[1].keys() == outputs[4].keys()
    assert outputs[1] == outputs[4]


def test_parallel_conversion_bounds_buffered_bytes(tmp_path, monkeypatch):
    source_dir = _sharded_source(tmp_path)
    serial = tmp_path / "serial.infernix"
    _convert_sharded(source_dir, serial, workers=1)
    expected = _object_bytes(serial)
    weight_bytes = {len(data) for key, data in expected.items() if key.startswith("weight/")}
    assert len(weight_bytes) == 1
    object_bytes = weight_bytes.pop()

    # Objects buffered at once, counted from the start of their conversion until the ordered
    # writer has written them (a lower bound of the window from submission).
    lock = threading.Lock()
    state = {"live": set(), "peak": 0}
    produce = pipeline._produce_buffered

    def tracked(job, obj):
        with lock:
            state["live"].add(obj.id)
            state["peak"] = max(state["peak"], len(state["live"]))
        time.sleep(0.01)  # give the other workers time to overlap
        return produce(job, obj)

    write_object = ArtifactWriter.write_object

    def tracked_write(self, object_id, data):
        write_object(self, object_id, data)
        with lock:
            state["live"].discard(object_id)

    monkeypatch.setattr(pipeline, "_produce_buffered", tracked)
    monkeypatch.setattr(ArtifactWriter, "write_object", tracked_write)

    # A budget below one object still converts, one object at a time; two objects' bytes admit two.
    for budget, peak in ((1, 1), (2 * object_bytes, 2)):
        state.update(live=set(), peak=0)
        output = tmp_path / f"budget-{budget}.infernix"
        report = _convert_sharded(source_dir, output, workers=4, max_buffered_bytes=budget)
        assert report["max_buffered_bytes"] == budget
        assert state["peak"] <= peak
        assert _object_bytes(output) == expected


def test_max_buffered_bytes_must_be_positive(tmp_path):
    model, recipe = _direct_model()

    with pytest.raises(ValueError, match="--max-buffered-bytes must be a positive integer"):
        convert(
            model,
            recipe,
            tmp_path / "invalid.infernix",
            device="cpu",
            max_buffered_bytes=0,
        )


def test_workers_must_be_positive(tmp_path):
    model, recipe = _direct_model()

    with pytest.raises(ValueError, match="--workers must be a positive integer"):
        convert(
            model,
            recipe,
            tmp_path / "invalid.infernix",
            device="cpu",
            workers=0,
        )


def test_parallel_workers_are_cpu_only(tmp_path, monkeypatch):
    model, recipe = _direct_model()

    # Validation happens before Recipe.prepare(), so no CUDA work is launched.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(
        ValueError,
        match="--workers > 1 is currently supported only with --device cpu",
    ):
        convert(
            model,
            recipe,
            tmp_path / "cuda.infernix",
            device="cuda",
            workers=2,
        )
