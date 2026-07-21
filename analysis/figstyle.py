"""Shared figure style for the working paper.

One palette, one type scale, one axis-title convention across every figure
in results/jr16, results/appendix and results/incidence. Presentation only:
nothing here touches data, weights or CSVs.

Palette and typography follow the PolicyEngine house style: primary blue
(#2C6496), teal accent (#39C6C0) and a grey ladder, a blue sequential ramp
(blue-light -> blue -> blue-pressed) and a grey-white-blue diverging map for
the scenario heatmaps (PolicyEngine convention: negative in grey, positive in
blue, white midpoint). The legacy categorical names (AQUA/YELLOW/VIOLET/RED)
are retained as aliases onto PolicyEngine hues so existing scripts keep
working.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- PolicyEngine brand palette (canonical hexes from policyengine-core) ---
BLUE = BLUE_PRIMARY = "#2C6496"
BLUE_LIGHT = "#D8E6F3"
BLUE_PRESSED = "#17354F"
BLUE_98 = "#F7FAFD"
TEAL = TEAL_ACCENT = "#39C6C0"
TEAL_PRESSED = "#227773"
DARKEST_BLUE = "#0C1A27"
GREEN = DARK_GREEN = "#558B2F"
DARK_GRAY = "#616161"
GRAY = "#808080"
MEDIUM_DARK_GRAY = "#D2D2D2"
LIGHT_GRAY = "#F2F2F2"

# Categorical slots (fixed order — never cycled), all PolicyEngine hues chosen
# for mutual separation.
SERIES = [BLUE, TEAL, GREEN, DARK_GRAY, DARKEST_BLUE, GRAY]

# Legacy aliases -> PolicyEngine hues (kept so existing scripts import cleanly).
# All six categorical names resolve to distinct, mutually separable brand hues.
AQUA = TEAL                  # #39C6C0
YELLOW = TEAL_PRESSED        # #227773 (distinct 6th hue, not a 2nd green)
VIOLET = DARKEST_BLUE        # #0C1A27
RED = GRAY                   # #808080

# Ink / chrome (PolicyEngine greys)
INK = DARKEST_BLUE            # near-black ink
INK2 = DARK_GRAY             # axis labels
MUTED = GRAY                 # tick labels
GRID = LIGHT_GRAY            # gridlines
BASELINE = MEDIUM_DARK_GRAY  # axis edges / baseline rule
NEUTRAL = LIGHT_GRAY         # neutral light fill
LIGHT_BLUE = BLUE_LIGHT      # sequential step (secondary bars)

# Sequential blue ramp (single-direction magnitudes).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "pe_seq", [BLUE_98, BLUE_LIGHT, BLUE, BLUE_PRESSED]
)

# Diverging map: grey (negative) — white — blue (positive), PolicyEngine
# convention (negative in grey, positive in blue).
DIVERGING = LinearSegmentedColormap.from_list(
    "pe_div",
    [DARK_GRAY, GRAY, MEDIUM_DARK_GRAY, "#FFFFFF", BLUE_LIGHT, BLUE, BLUE_PRESSED],
)

DECILE_AXIS = "Income decile (equivalised household disposable income, HBAI)"

# Canonical figure sizes (inches)
SINGLE = (8.0, 4.5)
HEATMAP = (10.0, 5.0)
FACETS = (16.0, 6.5)
TWOPANEL = (11.0, 4.5)
DPI = 200

# Serif family matching PolicyEngine's Roboto Serif; falls back cleanly to
# DejaVu Serif (bundled with matplotlib) when Roboto Serif is not installed.
_SERIF = ["Roboto Serif", "Roboto Slab", "Source Serif Pro", "DejaVu Serif"]


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": _SERIF,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def decile_ax(ax, ylabel, xlabel=DECILE_AXIS):
    """Common decile-chart chrome: x ticks 1-10, y grid only."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 11))
    ax.grid(axis="x", visible=False)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def legend_below(ax, ncol):
    """Legend centred below the axes (house style: legends never sit on data)."""
    ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -0.18))
