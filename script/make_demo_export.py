#!/usr/bin/env python3
"""Retired: the demo export is no longer generated here.

The shared demo study (``zotqda-fragments-demo.csv`` and
``zotqda-history-demo.csv``) used to be a synthetic file written by this
script.  It is now a *genuine* zotQDA export: the zotQDA repository builds a
native zotQDA model and drives the plugin's real exporters, so the CSVs are
produced by the same code that exports any real study.

Regenerate and redistribute from the zotQDA checkout instead::

    node tool/gen-demo-study.js --distribute

which drives zotQDA's real export code and copies the CSVs into this package
(and into qdaR) so the shipped copies stay byte-identical.  The same study is
also shipped as a native loadable example inside the zotQDA plugin, a REFI-QDA
project under ``examples/pflege-workload``.

This script now does nothing but say so, and exits non-zero, so nobody
regenerates the demo through the old synthetic path.
"""

import sys

MESSAGE = __doc__


def main() -> int:
    sys.stderr.write(MESSAGE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
