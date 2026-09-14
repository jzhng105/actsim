"""In-process artifact store: the handle layer between tools.

Simulations, fits and claim books are large; sending them through the model's
context is both slow and lossy. Instead each producing tool stores its object
here and returns a short, collision-free handle (``sim_7f3a1c2b``) plus a compact
summary. Downstream tools take that handle.

The store is bounded on two axes - number of artifacts and total stored cells -
and evicts least-recently-used entries, so a long agent session cannot exhaust
the host's memory.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd

from .errors import ActsimToolError
from .settings import SETTINGS

ArtifactKind = Literal["dataset", "fit", "simulation", "portfolio", "policies", "claims", "triangle"]

_PREFIXES: dict[str, str] = {
    "dataset": "data",
    "fit": "fit",
    "simulation": "sim",
    "portfolio": "port",
    "policies": "book",
    "claims": "clm",
    "triangle": "tri",
}


def _estimate_cells(payload: Any) -> int:
    """Rough element count, used to bound total store size."""
    if isinstance(payload, pd.DataFrame):
        return int(payload.shape[0] * max(payload.shape[1], 1))
    if isinstance(payload, (pd.Series, np.ndarray)):
        return int(np.asarray(payload).size)
    if isinstance(payload, dict):
        return sum(_estimate_cells(v) for v in payload.values())
    return 1


@dataclass
class Artifact:
    """A stored object plus the metadata tools need to describe it."""

    id: str
    kind: ArtifactKind
    label: str
    payload: Any
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    cells: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.id,
            "kind": self.kind,
            "label": self.label,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(self.created_at)),
            **self.meta,
        }


class ArtifactStore:
    """Thread-safe, bounded LRU store keyed by opaque handles."""

    def __init__(self) -> None:
        self._items: "OrderedDict[str, Artifact]" = OrderedDict()
        self._lock = threading.RLock()

    def put(
        self,
        kind: ArtifactKind,
        label: str,
        payload: Any,
        meta: dict[str, Any] | None = None,
    ) -> Artifact:
        """Store an object and return the artifact wrapper holding its handle."""
        artifact = Artifact(
            id=f"{_PREFIXES.get(kind, 'art')}_{secrets.token_hex(4)}",
            kind=kind,
            label=label,
            payload=payload,
            meta=dict(meta or {}),
            cells=_estimate_cells(payload),
        )
        with self._lock:
            self._items[artifact.id] = artifact
            self._items.move_to_end(artifact.id)
            self._evict()
        return artifact

    def get(self, artifact_id: str, *, kind: ArtifactKind | None = None) -> Artifact:
        """Fetch an artifact, refreshing its LRU position.

        Raises an error naming the live handles when the id is unknown, which is
        the common case after a server restart.
        """
        with self._lock:
            artifact = self._items.get(artifact_id)
            if artifact is None:
                known = ", ".join(self._items) or "none"
                raise ActsimToolError(
                    f"No artifact with id {artifact_id!r}.",
                    hint=(
                        "Artifacts live in server memory and are lost on restart. "
                        f"Currently stored: {known}. Call actsim_list_artifacts to inspect them, "
                        "or re-run the tool that produces this artifact kind."
                    ),
                )
            self._items.move_to_end(artifact_id)
        if kind is not None and artifact.kind != kind:
            raise ActsimToolError(
                f"Artifact {artifact_id!r} is a {artifact.kind}, but this tool needs a {kind}.",
                hint=f"Use actsim_list_artifacts to find an artifact of kind {kind!r}.",
            )
        return artifact

    def delete(self, artifact_id: str) -> bool:
        with self._lock:
            return self._items.pop(artifact_id, None) is not None

    def clear(self) -> int:
        with self._lock:
            count = len(self._items)
            self._items.clear()
            return count

    def __iter__(self) -> Iterator[Artifact]:
        with self._lock:
            return iter(list(self._items.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def total_cells(self) -> int:
        with self._lock:
            return sum(a.cells for a in self._items.values())

    def _evict(self) -> None:
        """Drop oldest entries until both bounds are satisfied. Caller holds lock."""
        while len(self._items) > SETTINGS.max_artifacts:
            self._items.popitem(last=False)
        while (
            sum(a.cells for a in self._items.values()) > SETTINGS.max_artifact_cells
            and len(self._items) > 1
        ):
            self._items.popitem(last=False)


STORE = ArtifactStore()
