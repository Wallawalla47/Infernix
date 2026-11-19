"""A drafter copied byte for byte from an existing v3 artifact (``--graft COMPONENT=ARTIFACT``).

The copied component keeps its stored objects, bindings and Uses exactly; the new artifact's own heads
read its final hidden state (``build_model``'s ``grafted`` records add that input, so the recipe's head
policy covers it). Its bindings and Uses must be exactly those of a drafter of its config over the new
target (``qwen3_5.draft_parameters``), so a drafter of another geometry is refused here rather than at
load. A drafter only proposes tokens the model verifies, so copying one changes acceptance, never output.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import prod

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, TensorSpec, binding_parts

from .model import Model
from .qwen3_5 import draft_parameters

GRAFTABLE = ("dflash", "dflash2")


def _object_ids(binding: dict) -> list[str]:
    if "object" in binding:
        return [binding["object"]]
    return [part["object"] for part in binding["parts"]]


@dataclass(frozen=True, slots=True)
class Graft:
    """One component's record, bindings, Uses and objects in an open source artifact (source ids)."""

    component: str
    artifact: Artifact
    record: dict
    bindings: dict
    uses: tuple[dict, ...]
    objects: tuple[str, ...]

    @classmethod
    def read(cls, artifact: Artifact, component: str) -> Graft:
        if component not in GRAFTABLE:
            raise ValueError(f"--graft copies a drafter ({', '.join(GRAFTABLE)}), not {component!r}")
        directory = artifact.directory
        if component not in directory.components:
            raise ValueError(f"{artifact.path}: no {component} component to graft")
        prefix = component + "/"
        bindings = {name: deepcopy(b) for name, b in directory.bindings.items() if name.startswith(prefix)}
        uses = tuple(deepcopy(u) for u in directory.uses if u["parameter"].startswith(prefix))
        for use in uses:
            if not use["input"].startswith(prefix):
                raise ValueError(f"{use['parameter']}@{use['input']}: a grafted drafter reads only its own inputs")
        objects = {object_id for binding in bindings.values() for object_id in _object_ids(binding)}
        objects |= {ref["object"] for use in uses for ref in (use.get("auxiliaries") or {}).values()}
        # Objects shared with the source's other parameters cannot move without them.
        for name, binding in directory.bindings.items():
            if not name.startswith(prefix) and objects.intersection(_object_ids(binding)):
                raise ValueError(f"{component}: shares an object with {name}; it cannot be grafted alone")
        for object_id in objects:
            if not isinstance(artifact.object(object_id), TensorObject):
                raise ValueError(f"{component}: {object_id} is not a tensor object")
        return cls(component, artifact, deepcopy(directory.components[component]), bindings, uses,
                   tuple(sorted(objects)))

    def check(self, model: Model) -> None:
        """The bindings and Uses are exactly those of this drafter over ``model``'s text."""
        if self.component not in model.grafted:
            raise ValueError(f"{self.component}: build the model with this graft's record")
        expected = draft_parameters(model.components[self.component]["config"], model.config, self.component)
        if set(self.bindings) != set(expected):
            missing = sorted(set(expected) - set(self.bindings))[:4]
            extra = sorted(set(self.bindings) - set(expected))[:4]
            raise ValueError(f"grafted {self.component}: bindings differ from its config (missing {missing}, "
                             f"extra {extra})")
        objects = self.artifact.by_id
        for name, parameter in expected.items():
            elements = sum(end - begin for _, begin, end in binding_parts(self.bindings[name], objects, name))
            if elements != prod(parameter.shape):
                raise ValueError(f"grafted {name}: {elements} elements, its shape {parameter.shape} needs "
                                 f"{prod(parameter.shape)}")
        declared = {(u["parameter"], u["input"]) for u in self.uses}
        wanted = {(name, i) for name, parameter in expected.items() for i in parameter.inputs}
        if declared != wanted or len(declared) != len(self.uses):
            raise ValueError(f"grafted {self.component}: its Uses differ from its parameters' inputs")

    def place(self, first: int) -> tuple[list[TensorSpec], dict, list[dict], list[tuple[str, str]]]:
        """Specs, bindings and Uses under new ids ``weight/{first...}``, and (new, source) id pairs."""
        ids = {old: f"weight/{first + i:06d}" for i, old in enumerate(self.objects)}
        specs = []
        for old, new in ids.items():
            obj = self.artifact.object(old)
            specs.append(TensorSpec(new, tuple(obj.shape), obj.format, obj.layout))
        bindings = {}
        for name, binding in self.bindings.items():
            binding = deepcopy(binding)
            if "object" in binding:
                binding["object"] = ids[binding["object"]]
            else:
                for part in binding["parts"]:
                    part["object"] = ids[part["object"]]
            bindings[name] = binding
        uses = []
        for use in self.uses:
            use = deepcopy(use)
            for ref in (use.get("auxiliaries") or {}).values():
                ref["object"] = ids[ref["object"]]
            uses.append(use)
        return specs, bindings, uses, [(new, old) for old, new in ids.items()]

    def provenance(self) -> dict:
        return {"artifact": str(self.artifact.path), "artifact_id": self.artifact.artifact_id.hex()}
