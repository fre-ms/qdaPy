"""The package version, in one place.

Its own module for two reasons.  `qdapy.reporting` stamps the version into a
generated checklist and would otherwise have to import the package that
imports it -- a cycle it used to dodge with an import inside the function
body.  And `pyproject.toml` reads the version from here through
`[tool.hatch.version]`, so the number is not written twice and cannot drift
between the metadata and what the package reports about itself.
"""

from __future__ import annotations

__version__ = "0.2.0"
