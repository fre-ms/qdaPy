# Changelog

## 0.1.0

First release.

* Reads all nine formats of the zotQDA exchange contract v1, validating the
  format stamp, the version and the promised columns; refuses an export from a
  newer version rather than guessing at it.
* Three plotting backends over one set of tables: `qdapy.gg` (plotnine),
  `qdapy.sns` (seaborn) and `qdapy.vega` (the plugin's own Vega-Lite charts,
  rendered unchanged).
* Intercoder reliability reimplemented independently of the plugin --
  percentage agreement, Cohen's, Fleiss', Brennan and Prediger's, Krippendorff's
  alpha, Gwet's AC1 -- plus level-wise agreement over a hierarchical code
  system and the confusion table. Checked against frozen plugin results.
* Inferential statistics the plugins omit: chi-squared with Cramér's V and an
  exact or Monte Carlo fallback, Jaccard code distances, classical scaling,
  hierarchical clustering with the cophenetic correlation, correspondence
  analysis with the share of inertia shown.
* Typed: the package ships `py.typed`, and the shapes it returns are declared
  rather than left as bare dictionaries -- `Segment` and `AlphaResult` in
  `qdapy.unitizing`, `Alignment` and `GammaResult` in `qdapy.gamma`,
  `Interval`, `Proportion` and `Paradox` in `qdapy.reliability`, `Saturation`
  in `qdapy.progress`, `Fit` in `qdapy.discovery`. `mypy` on the package is
  clean.
* The version lives in `qdapy/_version.py` and the packaging metadata reads it
  from there, so `qdapy.__version__` and what PyPI shows cannot drift apart.
* Licensed on the same terms as zotQDA and qdaZ: AGPL-3.0-or-later, with an
  additional permission under section 7 that places no condition on what you
  produce with it. A single licence, no proprietary exception and no
  contributor agreement to sign.
