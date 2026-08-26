# Contributing to qdaPy

Thank you for helping improve qdaPy. Bug reports, fixes and features are all
welcome.

## Build and test

qdaPy is a Python package (Python 3.11+). It uses [uv](https://docs.astral.sh/uv/).
The tests live in `test/`.

```sh
uv sync --extra vega --group dev   # install the package and its test tools
uv run pytest                      # the suite
uv run ruff check src test script  # style, incl. the C901 complexity gate
uv run mypy                        # the package ships py.typed: expected clean
```

The quality snapshot is a trend instrument, not a gate on every change, but you
can run it locally:

```sh
uv run python script/quality_metrics.py --baseline quality-baseline.json
```

## Proposing changes

- For anything larger than a small fix, open an issue first so the approach can
  be agreed before you write code.
- Keep pull requests focused: one topic per PR.
- Run `uv run pytest` before you submit, and add tests for new behaviour.
- Update the documentation (`doc/en`, `doc/de`) when behaviour changes.

## Licensing

qdaPy is licensed under the **AGPL-3.0-or-later**. Contributions are accepted
under the same licence. There is **no contributor licence agreement (CLA)** to
sign, and you **keep the copyright** to your contribution.
