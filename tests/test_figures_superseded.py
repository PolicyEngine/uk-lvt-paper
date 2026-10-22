"""The superseded fig9 chart is archived, not redrawn by default.

``results/figures/README.md`` labels ``fig9_net_change_by_region`` superseded:
it left the paper in PR #11. Its PNG and CSV stay in the repository, and
``analysis/figures.py`` redraws them only with ``--include-superseded``.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
FIG9 = "fig9_net_change_by_region"
sys.path.insert(0, str(ROOT / "analysis"))

import figures  # noqa: E402


@pytest.fixture
def drawn(monkeypatch):
    """Run ``figures.main`` without writing files; keep each figure's data."""
    saved = {}

    def keep(fig, df, stem):
        plt.close(fig)
        saved[stem] = df

    monkeypatch.setattr(figures, "_save", keep)
    return saved


def test_archived_fig9_files_are_kept():
    for suffix in (".png", ".csv"):
        assert (FIGURES / f"{FIG9}{suffix}").stat().st_size > 0
    assert "Superseded: `fig9_net_change_by_region`" in (
        (FIGURES / "README.md").read_text()
    )


def test_default_run_does_not_redraw_fig9(drawn):
    figures.main([])
    assert sorted(stem.split("_")[0] for stem in drawn) == [
        f"fig{i}" for i in range(1, 9)
    ]


def test_opt_in_redraws_the_archived_csv_from_the_committed_json(drawn):
    figures.main(["--include-superseded"])
    archived = (FIGURES / f"{FIG9}.csv").read_text()
    assert drawn[FIG9].to_csv(index=False, lineterminator="\n") == archived
