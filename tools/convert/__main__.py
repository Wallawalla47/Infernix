"""Convert explicitly selected local weights into a Infernix v3 artifact."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import sys
from collections.abc import Mapping

from .official_recipes import RECIPES
from .pipeline import DEFAULT_MAX_BUFFERED_BYTES, convert
from .proposal import DEFAULT_RANKING, add_official_proposal
from . import qwen4_exp
from .screen import add_screen_heads
from .qwen3_5 import build_model
from .recipe import Recipe
from .sources.safetensors import SafetensorsSource


def _open_named_source(path: Path):
    if path.suffix == ".gguf":
        from .sources.gguf import GGUFSource

        return GGUFSource(path)
    return SafetensorsSource(path)


class SourceInputs(Mapping):
    """Named optional sources are opened only when a recipe or component requests one."""

    def __init__(self, base, paths, stack):
        self._sources = {"base": base}
        self._paths = dict(paths)
        self._stack = stack

    def __getitem__(self, name):
        if name not in self._sources:
            if name not in self._paths:
                raise ValueError(
                    f"selected recipe requires source {name!r}; provide --source {name}=PATH"
                )
            self._sources[name] = self._stack.enter_context(
                _open_named_source(self._paths[name])
            )
        return self._sources[name]

    def __iter__(self):
        return iter(dict.fromkeys((*self._sources, *self._paths)))

    def __len__(self):
        return len(set(self._sources) | set(self._paths))

    def provenance(self):
        return {
            name: {"path": str(source.path)} for name, source in self._sources.items()
        }


def _pairs(values, label):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"{label} requires unique NAME=PATH entries")
        result[name] = Path(path)
    return result


def _recipe_parts(value: str):
    """Split a FILE[:function] recipe reference.

    Only a colon followed by a bare identifier is the separator; any other colon -
    such as a Windows drive letter, or one inside a POSIX path component - belongs
    to the path."""
    filename, separator, function = value.rpartition(":")
    if not separator or not function.isidentifier():
        return value, "configure"
    return filename, function


def _function(value: str):
    if value in RECIPES:
        return RECIPES[value]
    filename, function = _recipe_parts(value)
    path = Path(filename).resolve()
    spec = importlib.util.spec_from_file_location("infernix_user_recipe", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load recipe file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    result = getattr(module, function)
    if not callable(result):
        raise TypeError(f"{value}: recipe entry must be callable")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="primary checkpoint/config and default resources",
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help="official name or Python file[:function]; defaults to configure in a file",
    )
    parser.add_argument(
        "--override",
        help="Python configuration applied after the selected recipe and proposal",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="named source such as quantized, dflash or dflash2",
    )
    parser.add_argument(
        "--components",
        default="text",
        help="comma-separated text,vision,mtp,dflash,dflash2; dflash2 alone writes a Qwen4Exp "
        "DFlash2 companion artifact (--model: the target checkpoint)",
    )
    parser.add_argument(
        "--resource",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="override a final frontend resource",
    )
    parser.add_argument(
        "--proposal",
        action="store_true",
        help="include the existing indexed proposal head",
    )
    parser.add_argument("--proposal-rows", type=int, default=131072)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument(
        "--screen-heads",
        action="store_true",
        help="Qwen4Exp: add 2-bit screens (q2_g64_fp16) of the LM head and, with --proposal, of the "
        "proposal head, for the screened heads (tools/convert/screen.py); the artifact's last objects",
    )
    parser.add_argument("--name", help="public instance name saved in metadata")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel CPU weight conversion workers (--device cpu only); each of up to "
        "2 x WORKERS objects in flight is held whole in RAM, within --max-buffered-bytes",
    )
    parser.add_argument(
        "--max-buffered-bytes",
        type=int,
        default=DEFAULT_MAX_BUFFERED_BYTES,
        help="with --workers > 1: at most this many bytes of in-flight objects are buffered "
        "(one larger object still converts, alone); the workers' source reads and temporaries "
        f"come on top (default {DEFAULT_MAX_BUFFERED_BYTES})",
    )
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    parser.add_argument("--max-file-bytes", type=int, default=32_000_000_000)
    parser.add_argument(
        "--ngram-out",
        type=Path,
        help="Qwen4Exp n-gram volume (default: OUT + '.ngram'); it may live on another drive",
    )
    parser.add_argument(
        "--ngram-reuse",
        type=Path,
        help="Qwen4Exp: bind the artifact to this existing volume of the same checkpoint instead of writing one; "
        "a BF16 volume (tools.flash_next.ngram_bf16) binds the unquantized table, checked against the checkpoint's",
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        action="append",
        default=[],
        metavar="SHARD",
        help="import the weights bit-exactly from these GGUF shards (all of one split model): a Qwen4Exp "
        "GGUF, or a Qwen3.5 dense qwen35 GGUF (text only); --model then supplies the config, frontend "
        "resources and, for Qwen4Exp with the mtp component, the MTP head",
    )
    parser.add_argument(
        "--mmproj",
        type=Path,
        help="Qwen4Exp with --gguf and the vision component: the vision tower from this BF16 mmproj GGUF",
    )
    args = parser.parse_args(argv)
    components = tuple(args.components.split(","))
    if len(components) != len(set(components)):
        raise ValueError("components must not repeat")
    paths = _pairs(args.source, "source")
    if "base" in paths:
        raise ValueError("select the base source with --model")
    overrides = _pairs(args.resource, "resource")
    with ExitStack() as stack:
        base = stack.enter_context(SafetensorsSource(args.model))
        sources = SourceInputs(base, paths, stack)
        qwen4 = qwen4_exp.is_qwen4_exp(base.config)
        companion = "text" not in components
        if companion and (
            components != ("dflash2",)
            or not qwen4
            or args.gguf
            or args.mmproj
            or args.proposal
            or args.screen_heads
            or args.ngram_out
            or args.ngram_reuse
            or overrides
        ):
            raise ValueError(
                "without text, --components dflash2 writes a Qwen4Exp DFlash2 companion: --model is "
                "the target checkpoint and --source dflash2=PATH the drafter; no GGUF, proposal, "
                "n-gram or resource options apply"
            )
        gguf_store = None
        if args.gguf:
            from .sources.gguf_raw import GgufStore

            gguf_store = stack.enter_context(GgufStore(args.gguf))
        mmproj = None
        if args.mmproj is not None:
            if gguf_store is None or not qwen4:
                raise ValueError("--mmproj supplies the vision tower of a Qwen4Exp --gguf import")
            from .sources.gguf import GGUFSource

            mmproj = stack.enter_context(GGUFSource(args.mmproj))
        ngram_out = None
        if companion:
            model = qwen4_exp.build_dflash2_companion(base, sources["dflash2"])
        elif qwen4:
            # The checkpoint's own table format; a BF16 volume (tools/flash_next/ngram_bf16.py) binds the
            # table as its unquantized rows.
            ngram_format = None
            if args.ngram_reuse is not None:
                ngram_format = qwen4_exp.ngram_volume_format(base, args.ngram_reuse)
            if gguf_store is not None:
                if ngram_format == "bf16":
                    raise ValueError("a GGUF import binds its own IQ4_NL volume; give it a BF16 volume at run "
                                     "time with --ngram-volume")
                from . import qwen4_exp_gguf

                model = qwen4_exp_gguf.build_model(
                    base, gguf_store, components=components, resource_overrides=overrides, mmproj=mmproj
                )
                table_of = lambda: qwen4_exp_gguf.Iq4NlTable(gguf_store, model.config["ngram_table"]["rows"])
            else:
                model = qwen4_exp.build_model(
                    base, components=components, resource_overrides=overrides, ngram_format=ngram_format
                )
                table_of = lambda: qwen4_exp.ngram_table(base, model.config)
            # The volume id ties the separately placed n-gram volume to this artifact.
            if args.ngram_reuse is not None:
                if args.ngram_out is not None:
                    raise ValueError("--ngram-reuse and --ngram-out are exclusive")
                volume_id = qwen4_exp.read_ngram_volume_id(table_of(), args.ngram_reuse, bf16=ngram_format == "bf16")
                model.config["ngram_table"]["volume_id"] = volume_id.hex()
                ngram_out = None
            else:
                ngram_out = args.ngram_out or Path(str(args.out) + ".ngram")
                if ngram_out.exists():
                    raise FileExistsError(f"n-gram volume already exists: {ngram_out}")
                model.config["ngram_table"]["volume_id"] = os.urandom(16).hex()
        else:
            if args.ngram_out is not None:
                raise ValueError("--ngram-out applies only to Qwen4Exp sources")
            if gguf_store is not None:
                from . import qwen3_5_gguf

                model = qwen3_5_gguf.build_model(
                    base, gguf_store, components=components, resource_overrides=overrides
                )
            else:
                companions = {
                    key: sources[key] for key in ("dflash", "dflash2") if key in components
                }
                model = build_model(
                    base,
                    components=components,
                    companions=companions,
                    resource_overrides=overrides,
                )
        recipe = Recipe(model)
        _function(args.recipe)(model, recipe, sources)
        if args.proposal:
            add_official_proposal(recipe, ranking=args.ranking, rows=args.proposal_rows)
        if args.screen_heads:
            # After every other parameter: the copies are the artifact's last objects.
            if not qwen4 or gguf_store is not None:
                raise ValueError("--screen-heads screens a Qwen4Exp artifact converted from its checkpoint (not --gguf)")
            add_screen_heads(recipe)
        if args.override:
            _function(args.override)(model, recipe, sources)

        def progress(index, total, job):
            label = job.parameters[0]
            if len(job.parameters) > 1:
                label += f" (+{len(job.parameters)-1})"
            print(
                f"[{index+1}/{total}] {label}: {job.spec.format} {job.spec.shape}",
                flush=True,
            )

        provenance = {
            "converter": "infernix-v3",
            "recipe": args.recipe,
            "sources": sources.provenance(),
        }
        if gguf_store is not None:
            provenance["gguf"] = [str(path) for path in gguf_store.paths]
        if mmproj is not None:
            provenance["mmproj"] = str(mmproj.path)
        if args.override:
            provenance["override"] = args.override
        if args.proposal:
            provenance["ranking"] = str(args.ranking)
        if args.screen_heads:
            provenance["screen_heads"] = {"format": "q2_g64_fp16", "method": "grouped_mse_norm"}
        report = convert(
            model,
            recipe,
            args.out,
            name=args.name,
            provenance=provenance,
            device=args.device,
            rows_per_chunk=args.rows_per_chunk,
            workers=args.workers,
            max_buffered_bytes=args.max_buffered_bytes,
            max_file_bytes=args.max_file_bytes,
            progress=progress,
        )
        print(
            f"wrote {args.out}: {report['objects']} objects, {len(report['files'])} files, {report['seconds']:.1f}s",
            flush=True,
        )
        if qwen4 and ngram_out is not None:
            table = model.config["ngram_table"]

            def ngram_progress(index, total):
                print(f"[ngram {index+1}/{total}]", flush=True)

            qwen4_exp.write_ngram_volume(
                table_of(),
                ngram_out,
                bytes.fromhex(table["volume_id"]),
                progress=ngram_progress,
            )
            print(f"wrote {ngram_out}: {table['blocks']} blocks", flush=True)


if __name__ == "__main__":
    main()
