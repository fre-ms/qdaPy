"""The plugin's own charts, unchanged.

Every chart in qdaZ can be saved as a "data + spec" pair: the data as CSV, the
chart as a Vega-Lite specification that states its provenance in ``usermeta``.
Rendering that specification reproduces the figure exactly as it looked in
Zotero -- pixel for pixel, without anyone reimplementing it.

Use this when a figure has to match the plugin.  Use :mod:`qdapy.gg` or
:mod:`qdapy.sns` when it has to match the rest of a manuscript.

Altair is optional; install it with ``pip install "qdaPy[vega]"``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from .contract import contract

__all__ = ["PRODUCER", "read_spec", "render"]

PRODUCER = "qdaZ"


def read_spec(path: str | Path) -> dict[str, Any]:
    """Read a chart specification and check what it says about itself.

    The ``usermeta`` block names the producer, the contract version and the
    analysis.  A specification from a newer exchange version still renders --
    Vega-Lite does not care -- but the caller is warned, because the columns it
    refers to may have changed meaning.
    """
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    meta = spec.get("usermeta") or {}
    if meta.get("version") and meta["version"] > contract()["version"]:
        warnings.warn(
            f"specification states exchange version {meta['version']}, "
            f"qdaPy implements {contract()['version']}",
            stacklevel=2,
        )
    return spec


def render(spec: dict[str, Any] | str | Path,
           data: pd.DataFrame | None = None) -> Any:
    """Render a specification with Altair.

    Parameters
    ----------
    spec:
        A specification, or a path to one.
    data:
        Optional data to inline, e.g. the CSV saved next to the spec.  The
        plugin writes the two side by side precisely so that the figure can be
        rebuilt from the data rather than trusted as an image.
    """
    try:
        import altair as alt
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise ModuleNotFoundError(
            "rendering the plugin's own chart needs altair: "
            'pip install "qdaPy[vega]" -- or use qdapy.gg / qdapy.sns, which '
            "draw the same figures from the same tables"
        ) from exc

    if isinstance(spec, (str, Path)):
        spec = read_spec(spec)
    spec = dict(spec)
    if data is not None:
        spec["data"] = {"values": data.to_dict(orient="records")}
    return alt.Chart.from_dict(spec)
