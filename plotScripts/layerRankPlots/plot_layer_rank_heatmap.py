#!/usr/bin/env python3
"""
Heatmap: which (layer, rank) combo is most separable?

Writes two figures:
  • layer_rank_heatmap_appendix.png — four metrics in a 2×2 grid
  • layer_rank_heatmap_main.png     — composite only (main)
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

# Low → pink #99004D; mid stays tinted (no near-white bar); high → vivid blues.
PINK_LOW = "#99004D"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "separability_pink_blue",
    [
        (0.00, PINK_LOW),
        (0.18, "#B54878"),
        (0.35, "#C07090"),
        (0.48, "#A870A8"),
        (0.56, "#9078B8"),
        (0.64, "#7888CC"),
        (0.72, "#6888DC"),
        (0.80, "#5A82E8"),
        (0.88, "#4A7AEC"),
        (0.94, "#457EE8"),
        (1.00, "#5CA8FF"),
    ],
    N=256,
)
HEATMAP_CMAP.set_bad("#E8E8E8")


@dataclass(frozen=True)
class HeatmapTextSizes:
    suptitle: float
    panel_title: float
    axis_label: float
    tick: float
    cell: float
    cell_na: float
    cbar_tick: float
    rect_lw: float


TEXT_MAIN = HeatmapTextSizes(
    suptitle=15,
    panel_title=12,
    axis_label=15,
    tick=14,
    cell=14,
    cell_na=13,
    cbar_tick=13,
    rect_lw=3.0,
)
# Appendix: suptitle + per-panel titles + axis labels (“Layer”, “LoRA rank”) doubled
# again vs prior appendix; ticks / cells / colorbar unchanged.
TEXT_APPENDIX = HeatmapTextSizes(
    suptitle=92,
    panel_title=72,
    axis_label=77,
    tick=64,
    cell=66,
    cell_na=60,
    cbar_tick=62,
    rect_lw=6.5,
)


def _data_vmin_vmax(g: np.ndarray, mkey: str) -> tuple[float, float]:
    """
    Min/max actually present in the grid (finite values only).

    Used for Normalize and colorbar so the colormap spans the same interval as the
    data: widening the scale below/above the data made the colorbar show unused hues
    (e.g. pale band when ticks started at 0.2 but the smallest cell was 0.23).
    """
    valid = g[np.isfinite(g)]
    if valid.size == 0:
        return 0.0, 1.0
    lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        eps = max(abs(hi), 1.0) * 1e-4 + 1e-6
        return lo - eps, hi + eps

    if mkey in ("roc_auc_test", "probe_test_acc"):
        lo = max(0.5, lo)
        hi = min(1.0, hi)
        if hi <= lo:
            lo, hi = 0.5, 1.0
    elif mkey == "composite":
        lo = max(0.0, lo)
        hi = min(1.0, hi)
        if hi <= lo:
            lo, hi = 0.0, 1.0
    else:
        lo = max(0.0, lo)

    return lo, hi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT_DIR = ROOT / "resultsFinal" / "layerRankPlots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON = ROOT / "plotScripts" / "layerRankPlots" / "merged_probe_panel.json"
OUT_APPENDIX = OUT_DIR / "layer_rank_heatmap_appendix.png"
OUT_MAIN = OUT_DIR / "layer_rank_heatmap_main.png"

with open(JSON, encoding="utf-8") as f:
    data = json.load(f)

configs = data["configs"]

METRICS = [
    ("roc_auc_test",        "ROC-AUC",          True),   # (key, label, higher_is_better)
    ("probe_test_acc",      "Probe Acc",         True),
    ("kl_symmetric",        "Symmetric KL",      True),
    ("weight_spectral_diff","Spectral |ΔW|",     True),
]

# ── extract per (rank, paper_layer) ───────────────────────────────────────────
rows = []
seen = set()
for key, cfg in configs.items():
    rank        = cfg["rank"]
    layer_idx   = cfg["layer_idx"]
    paper_layer = cfg["paper_layer"]
    uid = (rank, paper_layer)
    if uid in seen:
        continue
    seen.add(uid)

    entry = {"rank": rank, "paper_layer": paper_layer}
    for mkey, _, _ in METRICS:
        arr = cfg.get(mkey, [])
        entry[mkey] = arr[layer_idx] if layer_idx < len(arr) else float("nan")
    rows.append(entry)

ranks  = sorted(set(r["rank"]        for r in rows))
layers = sorted(set(r["paper_layer"] for r in rows))
R, L   = len(ranks), len(layers)

def build_grid(mkey):
    g = np.full((R, L), np.nan)
    for r in rows:
        ri = ranks.index(r["rank"])
        li = layers.index(r["paper_layer"])
        g[ri, li] = r[mkey]
    return g

grids = {mkey: build_grid(mkey) for mkey, _, _ in METRICS}

# ── normalize each metric to [0,1] and average → composite ───────────────────
normed = []
for mkey, _, higher_better in METRICS:
    g = grids[mkey].copy()
    mn, mx = np.nanmin(g), np.nanmax(g)
    n = (g - mn) / (mx - mn + 1e-12) if mx > mn else g * 0 + 0.5
    if not higher_better:
        n = 1 - n
    normed.append(n)
composite = np.nanmean(np.stack(normed, axis=0), axis=0)

grids["composite"] = composite

APPENDIX_PANELS = list(METRICS)
MAIN_PANELS = [("composite", "Composite Score", True)]


def _colorbar_tick_label(v: float, _pos: int) -> str:
    """Whole number if exact (after 1-decimal round), else one decimal — never ×.××."""
    if not np.isfinite(v):
        return ""
    r = round(float(v), 1)
    ir = int(round(r))
    if abs(r - ir) < 1e-6:
        return str(ir)
    return f"{r:.1f}"


def _cbar_ticks_clipped_to_data(mkey: str, dmin: float, dmax: float) -> np.ndarray:
    """
    Grid-aligned ticks (0.1, or 1 for KL) inside [dmin, dmax].

    Except Spectral |ΔW|: always at least 3 ticks — merge data min/max with the
    grid list, then add the midpoint if still fewer than 3 (endpoints forced).
    Other metrics: no vmin/vmax labels unless they sit on the grid.
    """
    dmin, dmax = float(dmin), float(dmax)
    if not (np.isfinite(dmin) and np.isfinite(dmax)):
        return np.array([0.0, 1.0])
    if dmax < dmin:
        dmin, dmax = dmax, dmin
    if abs(dmax - dmin) < 1e-14:
        return np.array([dmin])

    if mkey == "kl_symmetric":
        step = 1.0
        lo = math.floor(dmin + 1e-9)
        hi = math.ceil(dmax - 1e-9)
    else:
        step = 0.1
        lo = math.floor(dmin / step + 1e-9) * step
        hi = math.ceil(dmax / step - 1e-9) * step

    n = int(round((hi - lo) / step)) + 1
    n = min(max(n, 2), 600)
    grid = np.array([lo + i * step for i in range(n)], dtype=float)
    tol = 1e-8
    sel = grid[(grid >= dmin - tol) & (grid <= dmax + tol)]
    on_grid = sorted({float(t) for t in sel})

    if mkey == "weight_spectral_diff":
        ticks = sorted({*on_grid, dmin, dmax})
        ticks = [t for t in ticks if dmin - tol <= t <= dmax + tol]
        if len(ticks) < 3:
            mid = 0.5 * (dmin + dmax)
            if not any(abs(mid - t) < 1e-7 for t in ticks):
                ticks.append(mid)
            ticks = sorted(ticks)
        return np.array(ticks, dtype=float)

    if len(on_grid) >= 1:
        return np.array(on_grid, dtype=float)
    mid = 0.5 * (dmin + dmax)
    return np.array([mid], dtype=float)


def _draw_one_heatmap_panel(
    ax: plt.Axes,
    mkey: str,
    label: str,
    _: bool,
    *,
    sizes: HeatmapTextSizes,
    show_panel_title: bool,
    cbar_fraction: float,
    cbar_pad: float,
    axis_tick_pad: float,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    g = grids[mkey]
    is_composite = mkey == "composite"

    vmin, vmax = _data_vmin_vmax(g, mkey)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    im = ax.imshow(g, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    if sizes.cbar_tick >= 48:
        _cb_len, _cb_w = 16.0, 2.0
    elif sizes.cbar_tick >= 24:
        _cb_len, _cb_w = 7.0, 1.35
    else:
        _cb_len, _cb_w = 5.0, 1.1
    cbar.ax.tick_params(
        labelsize=sizes.cbar_tick,
        length=_cb_len,
        width=_cb_w,
    )
    cbar.set_ticks(_cbar_ticks_clipped_to_data(mkey, vmin, vmax))
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(_colorbar_tick_label)
    )

    for ri in range(R):
        for li in range(L):
            v = g[ri, li]
            if not np.isnan(v):
                txt = (
                    f"{v:.2f}"
                    if mkey
                    in (
                        "roc_auc_test",
                        "probe_test_acc",
                        "composite",
                        "weight_spectral_diff",
                    )
                    or is_composite
                    else f"{v:.1f}"
                )
                frac = (v - vmin) / (vmax - vmin + 1e-12)
                color = "#f5f5f5" if frac > 0.62 else "#141414"
                ax.text(
                    li,
                    ri,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=sizes.cell,
                    fontweight="bold",
                    color=color,
                )
            else:
                ax.text(
                    li,
                    ri,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=sizes.cell_na,
                    color="#bbb",
                )

    ax.set_xticks(range(L))
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=sizes.tick)
    ax.set_yticks(range(R))
    ax.set_yticklabels([f"r={r}" for r in ranks], fontsize=sizes.tick)
    if show_xlabel:
        ax.set_xlabel("Layer", fontsize=sizes.axis_label, labelpad=0)
    if show_ylabel:
        ax.set_ylabel("LoRA rank", fontsize=sizes.axis_label)
    ax.tick_params(axis="both", labelsize=sizes.tick, pad=axis_tick_pad)

    if show_panel_title:
        star = " ★" if is_composite else ""
        ax.set_title(
            label + star,
            fontsize=sizes.panel_title,
            fontweight="bold" if is_composite else "normal",
        )

    # Removed best-cell boundary box for a cleaner look.


def _draw_heatmap_row(
    panels: list[tuple[str, str, bool]],
    figsize: tuple[float, float],
    suptitle: str,
    suptitle_y: float,
    *,
    sizes: HeatmapTextSizes = TEXT_MAIN,
    show_panel_title: bool = True,
    cbar_fraction: float = 0.06,
    cbar_pad: float = 0.03,
    layout_w_pad: float | None = None,
    layout_h_pad: float | None = None,
    axis_tick_pad: float = 4.0,
) -> plt.Figure:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    fig.set_constrained_layout_pads(
        w_pad=0.04 if layout_w_pad is None else layout_w_pad,
        h_pad=0.08 if layout_h_pad is None else layout_h_pad,
        rect=(0, 0, 1, 0.972),
    )

    for j, (ax, (mkey, label, _hb)) in enumerate(zip(axes, panels)):
        _draw_one_heatmap_panel(
            ax,
            mkey,
            label,
            _hb,
            sizes=sizes,
            show_panel_title=show_panel_title,
            cbar_fraction=cbar_fraction,
            cbar_pad=cbar_pad,
            axis_tick_pad=axis_tick_pad,
            show_ylabel=(j == 0),
            show_xlabel=True,
        )

    fig.suptitle(
        suptitle, fontsize=sizes.suptitle, fontweight="bold", y=suptitle_y
    )
    return fig


def _draw_appendix_2x2(
    panels: list[tuple[str, str, bool]],
    figsize: tuple[float, float],
    suptitle: str,
    suptitle_y: float,
    *,
    sizes: HeatmapTextSizes,
    cbar_fraction: float,
    cbar_pad: float,
    layout_w_pad: float | None,
    layout_h_pad: float | None,
    axis_tick_pad: float,
    layout_rect: tuple[float, float, float, float] = (0, 0, 1, 0.972),
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.04 if layout_w_pad is None else layout_w_pad,
        h_pad=0.08 if layout_h_pad is None else layout_h_pad,
        rect=layout_rect,
    )
    flat = axes.ravel()
    for i, (ax, (mkey, label, _hb)) in enumerate(zip(flat, panels)):
        row, col = divmod(i, 2)
        _draw_one_heatmap_panel(
            ax,
            mkey,
            label,
            _hb,
            sizes=sizes,
            show_panel_title=True,
            cbar_fraction=cbar_fraction,
            cbar_pad=cbar_pad,
            axis_tick_pad=axis_tick_pad,
            show_ylabel=(col == 0),
            show_xlabel=(row == 1),
        )
    fig.suptitle(
        suptitle, fontsize=sizes.suptitle, fontweight="bold", y=suptitle_y
    )
    return fig


# ── appendix: four metrics (2×2) ─────────────────────────────────────────────
fig_a = _draw_appendix_2x2(
    APPENDIX_PANELS,
    # Large canvas: wide for L3…L27 ticks + tall so each heatmap cell is readable.
    figsize=(44.0, 32.5),
    suptitle="Layer/rank separability analysis per metric",
    suptitle_y=1.008,
    sizes=TEXT_APPENDIX,
    cbar_fraction=0.045,
    cbar_pad=0.11,
    layout_w_pad=1.22,
    layout_h_pad=1.06,
    axis_tick_pad=28.0,
    layout_rect=(0, 0, 1, 0.979),
)
fig_a.savefig(OUT_APPENDIX, dpi=160, bbox_inches="tight")
plt.close(fig_a)
print(f"Saved → {OUT_APPENDIX}")

# ── main: composite only ──────────────────────────────────────────────────────
fig_m = _draw_heatmap_row(
    MAIN_PANELS,
    figsize=(4.2, 3.75),
    suptitle="Layer-rank separability",
    suptitle_y=1.02,
    sizes=TEXT_MAIN,
    show_panel_title=False,
    axis_tick_pad=1.0,
)
fig_m.savefig(OUT_MAIN, dpi=160, bbox_inches="tight", pad_inches=0.0)
plt.close(fig_m)
print(f"Saved → {OUT_MAIN}")
