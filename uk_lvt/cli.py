"""Command-line entry point: ``uk-lvt-build``.

Runs the canonical generator, ``uk_lvt.pipeline_direct`` (requires the
``simulation`` extra and the licensed Enhanced FRS dataset), and writes
``results/lvt_results.json``. Accepts the same ``--dataset``, ``--output``
and ``--uk-data-root`` options as ``python -m uk_lvt.pipeline_direct``.

The policyengine.py v4 pipeline in ``uk_lvt/pipeline.py`` is a cross-check
only; its poverty and Gini outputs use different conventions and must not
overwrite the committed results, so no entry point drives it.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from .pipeline_direct import main as pipeline_direct_main

    return pipeline_direct_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
