"""qdaPy -- analyse the qualitative coding that zotQDA and qdaZ export.

The two Zotero plugins write versioned CSV files and Vega-Lite chart
specifications described by a machine-readable contract.  qdaPy reads them,
checks them, redraws the figures in two grammars of graphics, and adds the
inferential statistics the plugins leave out on purpose.

    >>> import qdapy
    >>> frag = qdapy.read_fragments(qdapy.example("zotqda-fragments.csv"))
    >>> counts = qdapy.code_counts(frag)

Three plotting backends draw the same figures from the same tables:
:mod:`qdapy.gg` (plotnine), :mod:`qdapy.sns` (seaborn) and :mod:`qdapy.vega`
(the plugin's own charts, unchanged).  The tables themselves are in
:mod:`qdapy.frames` -- the numbers behind a figure are what goes into a paper.

The reliability coefficients in :mod:`qdapy.agreement` are a deliberate second
implementation of the plugin's own, checked against frozen plugin results, so
a figure in a methods section has been computed twice by two code bases that
share nothing but the contract.
"""

from __future__ import annotations

from types import ModuleType

from . import (
    agreement,
    discovery,
    frames,
    planning,
    progress,
    reliability,
    reporting,
    stats,
    unitizing,
)
from ._version import __version__
from .agreement import (
    ac1,
    alpha,
    brennan,
    confusion,
    flatten_path,
    fleiss,
    kappa,
    level_agreement,
    percent_agreement,
    units,
    units_binary,
)
from .contract import contract, example, formats, stamp_column
from .discovery import documents_for, saturation_index
from .frames import code_counts, code_matrix, saturation, timeline
from .gamma import gamma
from .planning import kappa_lower, plan_kappa, plan_themes, theme_power
from .progress import code_drift, new_codes, saturation_ratio
from .qdpx import QdpxProject, read_qdpx
from .read import (
    ContractError,
    apply_mapping,
    read,
    read_codebook,
    read_fragments,
    read_history,
    read_mapping,
    read_uncoded,
)
from .reliability import (
    agreement_by_code,
    bootstrap_ci,
    paradox,
    wilson,
)
from .reporting import coreq, coreq_markdown, srqr, srqr_markdown
from .stats import (
    ca,
    ca_points,
    chisq,
    cluster,
    code_distance,
    mds,
    mds_points,
)
from .unitizing import pk, segments, unitizing_alpha, window_diff

_BACKENDS = frozenset({"gg", "sns", "vega"})


def __getattr__(name: str) -> ModuleType:
    """Import a plotting backend only when it is used.

    matplotlib, seaborn and plotnine together take a noticeable moment to
    import.  Reading a file and computing a coefficient should not pay for a
    figure nobody asked for, so the backends load on first access -- after
    which ``qdapy.gg`` and friends behave like ordinary attributes.
    """
    if name in _BACKENDS:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """The exports plus the lazy backends.

    Without this, tab completion and :func:`dir` would hide ``gg``, ``sns``
    and ``vega`` until something had already imported them -- the names
    exist, they are just not in ``globals()`` yet.
    """
    return sorted(set(__all__) | _BACKENDS)

__all__ = [
    "ContractError",
    "QdpxProject",
    "__version__",
    "ac1",
    "agreement",
    "agreement_by_code",
    "alpha",
    "apply_mapping",
    "bootstrap_ci",
    "brennan",
    "ca",
    "ca_points",
    "chisq",
    "cluster",
    "code_counts",
    "code_distance",
    "code_drift",
    "code_matrix",
    "confusion",
    "contract",
    "coreq",
    "coreq_markdown",
    "discovery",
    "documents_for",
    "example",
    "flatten_path",
    "fleiss",
    "formats",
    "frames",
    "gamma",
    "gg",
    "kappa",
    "kappa_lower",
    "level_agreement",
    "mds",
    "mds_points",
    "new_codes",
    "paradox",
    "percent_agreement",
    "pk",
    "plan_kappa",
    "plan_themes",
    "planning",
    "progress",
    "read",
    "read_codebook",
    "read_fragments",
    "read_history",
    "read_mapping",
    "read_qdpx",
    "read_uncoded",
    "reliability",
    "reporting",
    "saturation",
    "saturation_index",
    "saturation_ratio",
    "segments",
    "sns",
    "srqr",
    "srqr_markdown",
    "stamp_column",
    "stats",
    "theme_power",
    "timeline",
    "unitizing",
    "unitizing_alpha",
    "units",
    "units_binary",
    "vega",
    "wilson",
    "window_diff",
]
