# qdaPy

[![CI](https://github.com/fre-ms/qdaPy/actions/workflows/ci.yml/badge.svg)](https://github.com/fre-ms/qdaPy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qdaPy)](https://pypi.org/project/qdaPy/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22109965.svg)](https://doi.org/10.5281/zenodo.22109965)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Project status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: AGPL-3.0](https://img.shields.io/github/license/fre-ms/qdaPy)](LICENSE)

Analyse the qualitative coding that the Zotero plugins
[zotQDA](https://zotqda.org) and qdaZ export.

Primarily the analysis half of the **zotQDA ecosystem** — that is where the
full range is available. It also reads **REFI-QDA** (`.qdpx`) projects, so a
study kept in MAXQDA, ATLAS.ti, NVivo, QDA Miner or Dedoose can borrow the
metrics those programs do not document: confidence intervals on every
agreement coefficient, the kappa paradox diagnostics, the reliability of the
segmentation itself, saturation as a reportable number, and sample-size
planning. A `.qdpx` supports a subset, reported on import.

```python
import qdapy

frag = qdapy.read_fragments("zotqda-fragments.csv")
qdapy.gg.frequencies(frag)      # plotnine
qdapy.sns.frequencies(frag)     # seaborn
```

The distribution and the import name are the same: PyPI normalises `qdaPy` to
`qdapy`.

```sh
pip install qdaPy          # pandas, scipy, plotnine, seaborn
pip install "qdaPy[vega]"  # plus altair, to render the plugin's own charts
```

Full documentation, including a section for anyone wanting to extend the
package: <https://qdapy.zotqda.org/>

## What it does

**Reads the exchange files and checks them.** Every file zotQDA writes carries
a stamp in its first column that says what kind of export it is and which
version of the exchange format it uses. qdaPy compares that stamp against the
contract file both sides ship (`qdapy.contract()`). A file that claims a newer
version than this package knows stops with an error instead of being guessed
at:

```python
qdapy.read_fragments("zotqda-codebook.csv")
# ContractError: expected a 'fragments' export but got 'codebook'
```

**Draws the plugin's figures in three ways, from one set of tables.** The
tables live in `qdapy.frames`; the backends only decide how they look.

| Backend | Import | Returns | Use it when |
|---|---|---|---|
| plotnine | `qdapy.gg` | `ggplot` | you think in the grammar of graphics, or want the figure to match the R twin |
| seaborn | `qdapy.sns` | matplotlib `Axes` | the figure belongs in an existing matplotlib layout |
| Vega-Lite | `qdapy.vega` | Altair chart | the figure has to match the plugin exactly |

All backends export the same set of functions — `frequencies`, `code_matrix`,
`timeline`, `saturation`, `mds`, `level_agreement`, `ca_map`, and `qdapy.sns`
adds a `dendrogram`. None of them calls `plt.show()`, so figures appear when
your script says so, not whenever the library feels like it.

**Recomputes the reliability figures independently.** Percentage agreement,
Cohen's and Fleiss' kappa, Brennan and Prediger's kappa, Krippendorff's alpha
and Gwet's AC1, computed here in Python, from the exported file alone. The
plugin computes the same coefficients in JavaScript and qdaR in R, so a figure
that ends up in a methods section has been produced three times by three code
bases that share nothing but the contract. The test suite checks this against
frozen plugin results on randomly generated coder matrices.

```python
u = qdapy.units(frag, uncoded=qdapy.read_uncoded("zotqda-uncoded.csv"))
qdapy.agreement.agreement(u)     # every measure side by side
qdapy.level_agreement(u)         # and where in the code system agreement is lost
```

Building that matrix involves two decisions, and qdaPy makes you take them
consciously instead of deciding behind your back. First: when both coders left
a segment uncoded, they agreed the segment was irrelevant, but the matrix only
knows about those segments if you pass the `uncoded` export. Second: when a
coder put several codes on one segment, there is no single value to compare, so
the segment is set aside and counted in `multi_set_aside`. Quote that count
next to the coefficient. A kappa that quietly dropped a tenth of the material
is not the kappa of your study.

**Adds the statistics the plugins leave out on purpose.** qdaZ sticks to
description and never runs a significance test, because a test invites claims
that many qualitative designs cannot carry. If your design does support one,
this is where you run it, and you pick it yourself:

| Function | Question |
|---|---|
| `qdapy.chisq()` | Are codes distributed independently of a grouping? With Cramér's V, an honest `expected_ok` flag, and an exact or Monte Carlo p-value when the expected counts are too small. |
| `qdapy.ca()` | Which codes and documents attract each other, and how much of the table's inertia am I actually seeing? |
| `qdapy.mds()` | A map of codes by the segments they share. |
| `qdapy.cluster()` | Groups of codes — with the cophenetic correlation, because a dendrogram always looks convincing. |

**Carries code identity through.** Codes get renamed, moved and merged while a
project matures. Every export therefore names each code twice: `code` holds the
path a person reads, `codeId` a stable identifier that stays put through all of
that.

```python
qdapy.units(frag, value="codeId")   # follows a code across revisions
```

If your analysis groups by the path, a code vanishes from it as soon as
somebody renames or moves that code in Zotero. Grouping by `codeId` survives
such housekeeping.

## Checking the package itself

```
python -m pytest                  # the suite
ruff check src tests scripts      # style, plus the C901 complexity gate
mypy                              # the package ships py.typed: expected clean

python script/quality_metrics.py --baseline quality-baseline.json
```

The last one is a trend instrument rather than a verdict. It records
complexity, maintainability, docstring and coverage figures, and fails only
when a gated number moves the wrong way against the committed baseline.
Absolute thresholds make a poor gate: Nagappan, Ball and Zeller (2006)
[10.1145/1134285.1134349](https://doi.org/10.1145/1134285.1134349) found no
metric set that fits every project and recommend calibrating against a
project's own history instead. `uv sync --group quality` installs the pinned
tools it needs.

## The R twin

[qdaR](https://qdar.zotqda.org/) is the same tool written for R. It
reads the same files, computes the same coefficients and draws the same
figures, over there with ggplot2. A shared set of frozen fixtures keeps both
packages honest: every release gets checked against the plugin's results and
against the other package. Two genuine bugs in the R code turned up exactly
this way.

## Licence

**AGPL-3.0-or-later** ([`LICENSE`](LICENSE)), the same terms as zotQDA, qdaZ
and the R twin qdaR.

**What you produce with qdaPy is yours.** Figures, tables, coefficients,
reports: the licence places no condition on any of it, by an additional
permission under section 7 of the AGPL
([`AGPL-ADDITIONAL-PERMISSION.md`](AGPL-ADDITIONAL-PERMISSION.md)). Strictly speaking that
changes nothing, a copyleft licence has never reached into a program's output
— but a figure in a submitted manuscript is not the place for a licensing
question, so it is written down.

**Commercial use does not need a commercial licence.** The AGPL does not forbid
it. What it asks is that a *modified version* you distribute, or let others use
over a network, comes with its source. That is the only condition, and it
holds for everyone: qdaPy is not dual-licensed, and no proprietary exception is
for sale.

**Contributions** are welcome and nothing has to be signed. There is no
contributor licence agreement, because there is no second licence that would
need one. You contribute under the AGPL and keep your copyright.
