#!/usr/bin/env python3
"""
Visualize "HACK" token appearance in SVD token-space projections.
Uses Plotly with the same visual style as the calibration histogram.

Reads plotScripts/hackTokenPlots/svd_token_analysis.json and produces 5 figures:
  1. Best HACK score per adapter
  2. Rank of first HACK token (inverted)
  3. Count + cumulative HACK score (dual subplot)
  4. Summary by poison rate (quad subplot)
  5. Side-by-side benign vs poison top-10 tokens
"""

import json
from pathlib import Path
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
JSON_PATH = ROOT / "plotScripts" / "hackTokenPlots" / "svd_token_analysis.json"
OUT_BASE_DIR = ROOT / "resultsFinal" / "hackTokenPlots"
OUT_DIR = OUT_BASE_DIR / "hackFigures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────
with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]

# ── HACK family detection ──────────────────────────────────────────────────
HACK_PATTERN = re.compile(r"\bhack\b", re.IGNORECASE)


def is_hack_token(tok: str) -> bool:
    clean = tok.strip()
    return bool(HACK_PATTERN.search(clean)) or clean.lower() in ("acked",)


def extract_rate(name: str) -> str:
    m = re.search(r"pr(\d+)", name)
    if m:
        return f"{m.group(1)}%"
    return "benign"


# ── extract metrics per adapter ────────────────────────────────────────────
MODULE    = "o_proj"
DIRECTION = 0  # u0

records = []

for adapter_name, info in results.items():
    atype = info["type"]
    rate  = extract_rate(adapter_name)

    analysis = info["analysis"]
    if MODULE not in analysis:
        continue

    mod       = analysis[MODULE]
    direction = mod["directions"][DIRECTION]
    top_pos   = direction["top_tokens_pos"]
    top_neg   = direction["top_tokens_neg"]

    hack_entries_pos = [(tok, score) for tok, score in top_pos if is_hack_token(tok)]

    best_hack_score  = max([sc for _, sc in hack_entries_pos], default=0.0)
    first_hack_rank  = 0
    for i, (tok, sc) in enumerate(top_pos, 1):
        if is_hack_token(tok):
            first_hack_rank = i
            break

    n_hack_in_top20 = len(hack_entries_pos)
    cum_hack_score  = sum(sc for _, sc in hack_entries_pos)
    energy          = mod["energy_ratio"]

    records.append({
        "name":             adapter_name,
        "type":             atype,
        "rate":             rate,
        "best_hack_score":  best_hack_score,
        "first_hack_rank":  first_hack_rank,
        "n_hack_top20":     n_hack_in_top20,
        "cum_hack_score":   cum_hack_score,
        "energy":           energy,
    })

# ── sort by rate (benign -> 1% -> 3% -> 5%) ──────────────────────────────
RATE_ORDER = {"benign": 0, "1%": 1, "3%": 2, "5%": 3}
records.sort(key=lambda r: (RATE_ORDER.get(r["rate"], 99), r["name"]))

# ══════════════════════════════════════════════════════════════════════════
# STYLE CONSTANTS  (matching the calibration-histogram reference)
# ══════════════════════════════════════════════════════════════════════════
FONT = "Times, serif"

# One color per rate; benign = grey, poison = teal gradient
FILL = {
    "benign": "rgba(128, 128, 128, 0.75)",
    "1%":     "rgba(80, 200, 200, 0.70)",
    "3%":     "rgba(0, 180, 180, 0.75)",
    "5%":     "rgba(0, 150, 150, 0.85)",
}
LINE = {
    "benign": "rgba(60,  60,  60,  0.9)",
    "1%":     "rgba(0, 170, 170, 0.9)",
    "3%":     "rgba(0, 140, 140, 0.9)",
    "5%":     "rgba(0, 110, 110, 0.9)",
}
TXT = {
    "benign": "rgba(60,  60,  60,  1.0)",
    "1%":     "rgba(0, 170, 170, 1.0)",
    "3%":     "rgba(0, 140, 140, 1.0)",
    "5%":     "rgba(0, 110, 110, 1.0)",
}
PAT = {"benign": ".", "1%": "-", "3%": "-", "5%": "-"}

HACK_FILL   = "rgba(200, 50, 50, 0.80)"
HACK_LINE   = "rgba(160, 30, 30, 0.90)"
ABSENT_FILL = "rgba(210, 210, 210, 0.70)"
ABSENT_LINE = "rgba(170, 170, 170, 0.90)"

RATE_KEYS = ["benign", "1%", "3%", "5%"]


def _sanitize(tok: str) -> str:
    """Replace non-ASCII / control chars so Plotly+kaleido can render them."""
    out = []
    for ch in tok:
        if ch == "\n":
            out.append("\\n")
        elif ord(ch) < 32 or ord(ch) > 126:
            out.append("?")
        else:
            out.append(ch)
    return "".join(out) or "(empty)"


def _legend_label(rate: str) -> str:
    return f"<b>Benign</b>" if rate == "benign" else f"<b>Poison {rate}</b>"


def _short(name: str, rate: str) -> str:
    m = re.search(r"(\d{3})", name)
    idx = m.group(1) if m else "?"
    return f"B-{idx}" if "benign" in name else f"P-{idx} ({rate})"


labels = [_short(r["name"], r["rate"]) for r in records]


def _axis(**kw):
    """Base axis style shared by every figure."""
    base = dict(
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.08)",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="rgba(0, 0, 0, 1.0)",
        zerolinewidth=2.5,
        showline=False,
        tickfont=dict(size=21, family=FONT, color="rgba(0, 0, 0, 0.9)"),
    )
    base.update(kw)
    return base


def _layout(**kw):
    """Base layout shared by every figure."""
    base = dict(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT, size=14),
        hovermode="x unified",
    )
    base.update(kw)
    return base


def _legend_box(**kw):
    base = dict(
        bgcolor="rgba(255, 250, 240, 0.85)",
        bordercolor="rgba(0, 0, 0, 0.25)",
        borderwidth=1,
        font=dict(size=14, family=FONT),
        itemsizing="constant",
    )
    base.update(kw)
    return base


def _save(fig, stem):
    """Save HTML always; save PNG if kaleido is installed."""
    html = OUT_DIR / stem.replace(".png", ".html")
    fig.write_html(str(html))
    try:
        png = OUT_DIR / stem
        fig.write_image(str(png), scale=2)
        print(f"[OK] {png}")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[OK] {html}  (PNG failed: {exc})")


# helper: one bar-trace per rate group
def _add_rate_bars(fig, x_vals, y_vals, rate_list, *,
                   text_fmt=".3f", show_legend=True, legendgroup_suffix="",
                   row=None, col=None):
    """Add one trace per rate key present in *rate_list*."""
    for rk in RATE_KEYS:
        idxs = [i for i, r in enumerate(rate_list) if r == rk]
        if not idxs:
            continue
        xs = [x_vals[i] for i in idxs]
        ys = [y_vals[i] for i in idxs]
        txt = []
        for v in ys:
            if v == 0:
                txt.append("")
            elif isinstance(text_fmt, str):
                txt.append(f"{v:{text_fmt}}")
            else:
                txt.append(str(v))

        kw_add = {}
        if row is not None:
            kw_add["row"] = row
            kw_add["col"] = col

        fig.add_trace(go.Bar(
            x=xs, y=ys,
            name=_legend_label(rk),
            marker=dict(
                color=FILL[rk],
                line=dict(color=LINE[rk], width=1.5),
                pattern=dict(shape=PAT[rk], fillmode="overlay",
                             size=4, solidity=0.3, fgcolor=LINE[rk]),
            ),
            text=txt,
            textposition="outside",
            textfont=dict(size=12, color=TXT[rk], family=FONT),
            opacity=0.85,
            showlegend=show_legend,
            legendgroup=rk + legendgroup_suffix,
        ), **kw_add)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Best HACK score per adapter
# ══════════════════════════════════════════════════════════════════════════
fig1 = go.Figure()
_add_rate_bars(fig1, labels,
               [r["best_hack_score"] for r in records],
               [r["rate"] for r in records],
               text_fmt=".3f")

mx = max((r["best_hack_score"] for r in records), default=0.1)
fig1.update_layout(**_layout(
    title=dict(
        text="<b>Payload Signal: Best 'HACK' Token Score in o_proj u<sub>0</sub></b>",
        font=dict(size=21, family=FONT, color="rgba(0,0,0,0.95)"),
        x=0.5, xanchor="center", pad=dict(b=5, t=5)),
    xaxis=_axis(title=dict(text="Adapter",
                            font=dict(size=21, family=FONT, color="rgba(0,0,0,0.9)"),
                            standoff=5),
                zeroline=False),
    yaxis=_axis(title=dict(text="Score of best HACK-family token",
                            font=dict(size=21, family=FONT, color="rgba(0,0,0,0.9)"),
                            standoff=5),
                range=[0, mx * 1.18]),
    barmode="overlay", bargap=0.15,
    legend=_legend_box(orientation="v", yanchor="top", y=0.95,
                       xanchor="right", x=0.98),
    width=1000, height=450,
    margin=dict(l=60, r=35, t=60, b=90),
))

_save(fig1, "hack_best_score.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Average rank of first HACK token by poison rate (line graph)
#   All adapters used; absent = rank 21.
# ══════════════════════════════════════════════════════════════════════════
NOT_FOUND = 21

LINE2_COLOR = {
    "benign": "rgba(100, 140, 220, 1.0)",   # blue (DFE6FF, brightened)
    "1%":     "rgba(255, 180, 220, 1.0)",   # mild pink (FFE6F4, brightened)
    "3%":     "rgba(200,  80, 140, 1.0)",   # pink (between FFE6F4 and 99004D)
    "5%":     "rgba(153,   0,  77, 1.0)",   # deep pink (99004D)
}
LINE2_LABELS = {
    "benign": "Benign",
    "1%":     "Poison 1%",
    "3%":     "Poison 3%",
    "5%":     "Poison 5%",
}
X_ORDER = ["benign", "1%", "3%", "5%"]
X_LABELS = ["Benign", "Poison 1%", "Poison 3%", "Poison 5%"]
# Strongly compressed x-spacing so categories are almost adjacent.
X_NUMERIC = [0.0, 0.55, 1.10, 1.65]

# Compute avg rank per group
avg_ranks, point_labels = [], []
for rk in X_ORDER:
    group = [r for r in records if r["rate"] == rk]
    rank_vals_grp = [
        r["first_hack_rank"] if r["first_hack_rank"] > 0 else NOT_FOUND
        for r in group
    ]
    avg = np.mean(rank_vals_grp) if group else NOT_FOUND
    avg_ranks.append(avg)
    all_nf = all(r["first_hack_rank"] == 0 for r in records if r["rate"] == rk)
    point_labels.append("None" if all_nf else f"{avg:.1f}")

fig2 = go.Figure()

# ── line ──────────────────────────────────────────────────────────────────
fig2.add_trace(go.Scatter(
    x=X_NUMERIC, y=avg_ranks,
    mode="lines",
    line=dict(color="rgba(80,80,80,0.6)", width=2, dash="solid"),
    showlegend=False,
    hoverinfo="skip",
))

# ── coloured markers + labels ─────────────────────────────────────────────
for i, rk in enumerate(X_ORDER):
    fig2.add_trace(go.Scatter(
        x=[X_NUMERIC[i]], y=[avg_ranks[i]],
        mode="markers+text",
        name=f"<b>{LINE2_LABELS[rk]}</b>",
        marker=dict(size=26, color=LINE2_COLOR[rk],
                    line=dict(color="white", width=1.8)),
        text=[point_labels[i]],
        textposition="top center",
        textfont=dict(size=13, family=FONT, color=LINE2_COLOR[rk]),
        hovertemplate=f"<b>{LINE2_LABELS[rk]}</b><br>Avg rank: {avg_ranks[i]:.2f}<extra></extra>",
    ))

fig2.add_annotation(
    xref="paper", x=0.98, yref="y", y=18.5,
    text="Not in top-20", showarrow=False,
    xanchor="right", yanchor="top",
    font=dict(size=16, family=FONT, color="rgba(200,0,0,0.4)"),
)

fig2.update_layout(**_layout(
    title=dict(
        text="<b>Payload Visibility: Avg Rank of First 'HACK' Token<br>in o_proj u<sub>0</sub> Top-20</b>",
        font=dict(size=19, family=FONT, color="rgba(0,0,0,0.95)"),
        x=0.5, xanchor="center", pad=dict(b=5, t=5)),
    xaxis=_axis(
        title=dict(text="Poison Rate",
                   font=dict(size=19, family=FONT, color="rgba(0,0,0,0.9)"),
                   standoff=8),
        tickvals=X_NUMERIC, ticktext=X_LABELS,
        zeroline=False, range=[-0.10, 1.75],
        showline=True, linecolor="rgba(0,0,0,0.95)", linewidth=1.1, mirror=True,
    ),
        yaxis=_axis(
        title=dict(text="Avg rank of first HACK token",
                   font=dict(size=19, family=FONT, color="rgba(0,0,0,0.9)"),
                   standoff=5),
        range=[22, 0], dtick=2,
        zeroline=False,
        showline=True, linecolor="rgba(0,0,0,0.95)", linewidth=1.1, mirror=True,
    ),
    shapes=[
        dict(type="line", xref="paper", x0=0, x1=1,
             yref="y", y0=22, y1=22,
             line=dict(color="rgba(0,0,0,1.0)", width=2.5)),
        dict(type="line", xref="paper", x0=0, x1=1,
             yref="y", y0=20, y1=20,
             line=dict(color="rgba(200,0,0,0.22)", width=1.6, dash="dash")),
    ],
    showlegend=False,
    width=540, height=500,
    margin=dict(l=75, r=40, t=65, b=40),
))

_save(fig2, "hack_rank_position.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Count of HACK tokens + cumulative score  (two subplots)
# ══════════════════════════════════════════════════════════════════════════
fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "<b>Count of HACK Variants in Top-20</b>",
        "<b>Cumulative HACK Score in Top-20</b>",
    ],
    horizontal_spacing=0.12,
)

metrics3 = [
    ([r["n_hack_top20"]  for r in records], "d"),
    ([r["cum_hack_score"] for r in records], ".3f"),
]

for col_idx, (vals, fmt) in enumerate(metrics3, 1):
    for rk in RATE_KEYS:
        idxs = [i for i, r in enumerate(records) if r["rate"] == rk]
        if not idxs:
            continue
        xs = [labels[i] for i in idxs]
        ys = [vals[i] for i in idxs]
        txt = [f"{v:{fmt}}" if v > 0 else "" for v in ys]

        fig3.add_trace(go.Bar(
            x=xs, y=ys,
            name=_legend_label(rk),
            marker=dict(
                color=FILL[rk],
                line=dict(color=LINE[rk], width=1.5),
                pattern=dict(shape=PAT[rk], fillmode="overlay",
                             size=4, solidity=0.3, fgcolor=LINE[rk]),
            ),
            text=txt, textposition="outside",
            textfont=dict(size=12, color=TXT[rk], family=FONT),
            opacity=0.85,
            showlegend=(col_idx == 1),
            legendgroup=rk,
        ), row=1, col=col_idx)

fig3.update_layout(**_layout(
    title=dict(
        text="<b>HACK Token Presence in o_proj u<sub>0</sub> Top-20 Positive Direction</b>",
        font=dict(size=21, family=FONT, color="rgba(0,0,0,0.95)"),
        x=0.5, xanchor="center", pad=dict(b=5, t=5)),
    barmode="overlay", bargap=0.15,
    legend=_legend_box(orientation="v", yanchor="top", y=0.95,
                       xanchor="right", x=0.98),
    width=1200, height=470,
    margin=dict(l=60, r=35, t=80, b=90),
))
for c in (1, 2):
    fig3.update_xaxes(**_axis(zeroline=False), row=1, col=c)
    fig3.update_yaxes(**_axis(), row=1, col=c)
fig3.update_yaxes(title_text="# HACK tokens",  row=1, col=1)
fig3.update_yaxes(title_text="Sum score",       row=1, col=2)
for ann in fig3.layout.annotations:
    ann.font = dict(size=21, family=FONT, color="rgba(0,0,0,0.9)")

_save(fig3, "hack_count_and_cumulative.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Summary by poison rate (averaged, 4 subplots)
# ══════════════════════════════════════════════════════════════════════════
grouped = defaultdict(list)
for r in records:
    grouped[r["rate"]].append(r)

avg_best   = [np.mean([r["best_hack_score"] for r in grouped[k]]) for k in RATE_KEYS]
avg_count  = [np.mean([r["n_hack_top20"]    for r in grouped[k]]) for k in RATE_KEYS]
avg_cum    = [np.mean([r["cum_hack_score"]  for r in grouped[k]]) for k in RATE_KEYS]
avg_energy = [np.mean([r["energy"]          for r in grouped[k]]) for k in RATE_KEYS]

panel_info = [
    ("Avg Best HACK Score",          avg_best,   ".4f", None),
    ("Avg # HACK Tokens in Top-20",  avg_count,  ".1f", None),
    ("Avg Cumulative HACK Score",    avg_cum,    ".4f", None),
    ("Avg o_proj E(sigma_1)",        avg_energy, ".3f", [0.7, 1.02]),
]

fig4 = make_subplots(
    rows=1, cols=4,
    subplot_titles=[f"<b>{p[0]}</b>" for p in panel_info],
    horizontal_spacing=0.07,
)

for col_idx, (_, vals, fmt, yrange) in enumerate(panel_info, 1):
    for i, rk in enumerate(RATE_KEYS):
        fig4.add_trace(go.Bar(
            x=[rk], y=[vals[i]],
            name=_legend_label(rk),
            marker=dict(
                color=FILL[rk],
                line=dict(color=LINE[rk], width=1.5),
                pattern=dict(shape=PAT[rk], fillmode="overlay",
                             size=4, solidity=0.3, fgcolor=LINE[rk]),
            ),
            text=[f"{vals[i]:{fmt}}"],
            textposition="outside",
            textfont=dict(size=12, color=TXT[rk], family=FONT),
            opacity=0.85,
            showlegend=(col_idx == 1),
            legendgroup=rk,
        ), row=1, col=col_idx)

    if yrange:
        fig4.update_yaxes(range=yrange, row=1, col=col_idx)

fig4.update_layout(**_layout(
    title=dict(
        text="<b>SVD Token Analysis Summary by Poison Rate (o_proj, u<sub>0</sub>, Layer 20)</b>",
        font=dict(size=21, family=FONT, color="rgba(0,0,0,0.95)"),
        x=0.5, xanchor="center", pad=dict(b=5, t=5)),
    barmode="overlay", bargap=0.25,
    legend=_legend_box(orientation="v", yanchor="top", y=0.93,
                       xanchor="right", x=0.99, font=dict(size=21, family=FONT)),
    width=1250, height=470,
    margin=dict(l=50, r=35, t=85, b=50),
))
for c in range(1, 5):
    fig4.update_xaxes(**_axis(zeroline=False), row=1, col=c)
    fig4.update_yaxes(**_axis(), row=1, col=c)
for ann in fig4.layout.annotations:
    ann.font = dict(size=12, family=FONT, color="rgba(0,0,0,0.9)")

_save(fig4, "hack_summary_by_rate.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Side-by-side benign vs poison (5%) top-10 tokens
#   Uses dual-domain layout (no make_subplots) for kaleido compatibility.
# ══════════════════════════════════════════════════════════════════════════
benign_rec   = next(r for r in records if r["type"] == "benign")
poison5_recs = [r for r in records if r["rate"] == "5%"]


def _top_tokens(adapter_name, module="o_proj", direction=0, n=10):
    d = results[adapter_name]["analysis"][module]["directions"][direction]
    return d["top_tokens_pos"][:n]


b_name = benign_rec["name"]
b_toks = _top_tokens(b_name)
p5     = max(poison5_recs, key=lambda r: r["best_hack_score"])
p_name = p5["name"]
p_toks = _top_tokens(p_name)

# Prepare data (reversed so #1 is at top of horizontal chart)
b_labels_raw = [tok for tok, _ in b_toks][::-1]
b_labels = [_sanitize(t) for t in b_labels_raw]
b_scores = [sc for _, sc in b_toks][::-1]

p_labels_raw = [tok for tok, _ in p_toks][::-1]
p_labels = [_sanitize(t) for t in p_labels_raw]
p_scores = [sc for _, sc in p_toks][::-1]

# Use numeric y positions + custom tick labels to avoid categorical issues
b_y = list(range(len(b_labels)))
p_y = list(range(len(p_labels)))

b_colors = [HACK_FILL if is_hack_token(t) else FILL["benign"] for t in b_labels_raw]
b_lines  = [HACK_LINE if is_hack_token(t) else LINE["benign"] for t in b_labels_raw]
p_colors = [HACK_FILL if is_hack_token(t) else FILL["5%"] for t in p_labels_raw]
p_lines  = [HACK_LINE if is_hack_token(t) else LINE["5%"] for t in p_labels_raw]

fig5 = go.Figure()

# Benign bars → left panel (xaxis, yaxis)
fig5.add_trace(go.Bar(
    y=b_y, x=b_scores, orientation="h",
    name="<b>Benign token</b>",
    marker=dict(
        color=b_colors,
        line=dict(color=b_lines, width=1.5),
        pattern=dict(shape=".", fillmode="overlay", size=4, solidity=0.3,
                     fgcolor="rgba(60,60,60,0.4)"),
    ),
    text=[f"{s:.4f}" for s in b_scores],
    textposition="outside",
    textfont=dict(size=9, color=TXT["benign"], family=FONT),
    opacity=0.85,
    xaxis="x", yaxis="y",
    showlegend=True, legendgroup="benign_tok",
))

# Poison bars → right panel (xaxis2, yaxis2)
fig5.add_trace(go.Bar(
    y=p_y, x=p_scores, orientation="h",
    name="<b>Poison token</b>",
    marker=dict(
        color=p_colors,
        line=dict(color=p_lines, width=1.5),
        pattern=dict(shape="-", fillmode="overlay", size=5, solidity=0.3,
                     fgcolor="rgba(0,140,140,0.4)"),
    ),
    text=[f"{s:.4f}" for s in p_scores],
    textposition="outside",
    textfont=dict(size=9, color=TXT["5%"], family=FONT),
    opacity=0.85,
    xaxis="x2", yaxis="y2",
    showlegend=True, legendgroup="poison_tok",
))

# HACK legend marker (invisible bar for swatch)
fig5.add_trace(go.Bar(
    y=[None], x=[None], orientation="h",
    name="<b>HACK family</b>",
    marker=dict(color=HACK_FILL, line=dict(color=HACK_LINE, width=1.5)),
    showlegend=True,
))

max_score = max(max(b_scores), max(p_scores)) * 1.35

fig5.update_layout(**_layout(
    title=dict(
        text="<b>Side-by-Side: Token Projections Reveal Payload in Poisoned Adapters</b>",
        font=dict(size=21, family=FONT, color="rgba(0,0,0,0.95)"),
        x=0.5, xanchor="center", pad=dict(b=5, t=5)),
    # Left panel axes
    xaxis=dict(
        domain=[0, 0.44],
        title=dict(text="Projection Score",
                   font=dict(size=12, family=FONT, color="rgba(0,0,0,0.9)")),
        range=[0, max_score],
        **{k: v for k, v in _axis().items() if k != "zeroline"},
        zeroline=True,
    ),
    yaxis=dict(
        tickvals=b_y, ticktext=b_labels,
        **{k: v for k, v in _axis(zeroline=False).items()},
    ),
    # Right panel axes
    xaxis2=dict(
        domain=[0.56, 1.0],
        title=dict(text="Projection Score",
                   font=dict(size=12, family=FONT, color="rgba(0,0,0,0.9)")),
        range=[0, max_score],
        anchor="y2",
        **{k: v for k, v in _axis().items() if k != "zeroline"},
        zeroline=True,
    ),
    yaxis2=dict(
        tickvals=p_y, ticktext=p_labels,
        anchor="x2",
        **{k: v for k, v in _axis(zeroline=False).items()},
    ),
    barmode="overlay",
    legend=_legend_box(orientation="h", yanchor="bottom", y=-0.15,
                       xanchor="center", x=0.5,
                       font=dict(size=21, family=FONT)),
    width=1100, height=520,
    margin=dict(l=100, r=70, t=80, b=80),
    # Subplot title annotations
    annotations=[
        dict(text=f"<b>BENIGN: {_short(b_name, 'benign')} -- o_proj u0 Top-10</b>",
             xref="paper", yref="paper", x=0.22, y=1.04,
             showarrow=False,
             font=dict(size=21, family=FONT, color="rgba(0,0,0,0.9)")),
        dict(text=f"<b>POISON 5%: {_short(p_name, '5%')} -- o_proj u0 Top-10</b>",
             xref="paper", yref="paper", x=0.78, y=1.04,
             showarrow=False,
             font=dict(size=21, family=FONT, color="rgba(0,0,0,0.9)")),
    ],
))

_save(fig5, "hack_benign_vs_poison_top10.png")


print(f"\n[Done] All figures saved to {OUT_DIR}")
