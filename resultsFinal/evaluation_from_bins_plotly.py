import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Paste your binned eval hist data here (bin_start, bin_end, count, wrong_classified_count)
# If a bin has no "(wrong classified)" note, set wrong_classified_count=0.

BENIGN_BINS = [
    (31.985, 34.6567, 4, 0),
    (42.824, 45.4961, 2, 0),
    (45.4961, 48.1678, 1, 0),
    (48.1678, 50.8395, 5, 0),
    (50.8395, 53.5112, 5, 0),
    (53.5112, 56.1829, 6, 0),
    (56.1829, 58.8546, 8, 0),
    (58.8546, 61.5263, 4, 0),
    (61.5263, 64.1980, 2, 0),
    (64.1980, 66.8697, 7, 0),
    (66.8697, 69.5414, 5, 0),
    (69.5414, 73.2131, 1, 1),  # wrong classified (was 69.5414,72.2131 but now it wasnt almost vissible on the graph)
]

POISON_BINS = [
    (66.0305, 67.7862, 1, 1),  # wrong classified
    (69.4656, 71.2213, 1, 1),  # wrong classified
    (71.2213, 72.9770, 5, 0),
    (72.9770, 74.7327, 2, 0),
    (74.7327, 76.4884, 2, 0),
    (76.4884, 78.2441, 4, 0),
    (80.15267, 81.90837, 8, 0),
    (81.90837, 83.66407, 7, 0),
    (85.41985, 87.1755, 2, 0),
    (87.1755, 88.93125, 9, 0),
    (88.93125, 90.68695, 4, 0),
    (90.68695, 92.4426, 5, 0),
]


# Threshold (use 71.79 if your axis is 0–100; use 0.7179 if your axis is 0–1)
THRESHOLD = 71.79


def _maybe_rescale_to_unit_interval(
    bins: list[tuple[float, float, int, int]],
    threshold: float,
):
    """
    If values look like 0–100, rescale to 0–1.
    Heuristic: if any edge > 1.5, treat as percent-like.
    """
    max_edge = max(b for a, b, _, _ in bins)
    if max_edge > 1.5:
        scaled_bins = [(a / 100.0, b / 100.0, c, w) for a, b, c, w in bins]
        return scaled_bins, threshold / 100.0, True
    return bins, threshold, False


def _to_bar_arrays(bins: list[tuple[float, float, int, int]]):
    centers = []
    widths = []
    counts = []
    wrong = []
    for a, b, c, w in bins:
        centers.append((a + b) / 2.0)
        widths.append(max(b - a, 0.0))
        counts.append(c)
        wrong.append(w)
    return centers, widths, counts, wrong


def _expand_midpoints(bins: list[tuple[float, float, int, int]]):
    """
    Approximate raw samples for boxplots by expanding bin midpoints.
    This is NOT exact (original within-bin positions are lost).
    """
    vals: list[float] = []
    for a, b, c, _w in bins:
        mid = (a + b) / 2.0
        vals.extend([mid] * c)
    return vals


def main():
    benign_bins, threshold, scaled = _maybe_rescale_to_unit_interval(BENIGN_BINS, THRESHOLD)
    poison_bins, _, _ = _maybe_rescale_to_unit_interval(POISON_BINS, THRESHOLD)

    # X axis range: show full 0–1 if scaled, otherwise 0–100
    x_range = [0, 1] if scaled else [0, 100]

    bx, bw, bcounts, bwrong = _to_bar_arrays(benign_bins)
    px, pw, pcounts, pwrong = _to_bar_arrays(poison_bins)

    benign_total = sum(bcounts)
    poison_total = sum(pcounts)
    benign_wrong = sum(bwrong)
    poison_wrong = sum(pwrong)

    benign_acc = 0.0 if benign_total == 0 else (benign_total - benign_wrong) / benign_total
    poison_acc = 0.0 if poison_total == 0 else (poison_total - poison_wrong) / poison_total

    # Separate bars by position relative to threshold for proper layering
    # Benign: correct (below threshold) vs wrong (above threshold)
    benign_correct_x, benign_correct_w, benign_correct_y = [], [], []
    benign_wrong_x, benign_wrong_w, benign_wrong_y = [], [], []
    for i, (x, w, y) in enumerate(zip(bx, bw, bcounts)):
        if x < threshold:
            benign_correct_x.append(x)
            benign_correct_w.append(w)
            benign_correct_y.append(y)
        else:
            benign_wrong_x.append(x)
            benign_wrong_w.append(w)
            benign_wrong_y.append(y)
    
    # Poison: correct (above threshold) vs wrong (below threshold)
    poison_correct_x, poison_correct_w, poison_correct_y = [], [], []
    poison_wrong_x, poison_wrong_w, poison_wrong_y = [], [], []
    for i, (x, w, y) in enumerate(zip(px, pw, pcounts)):
        if x > threshold:
            poison_correct_x.append(x)
            poison_correct_w.append(w)
            poison_correct_y.append(y)
        else:
            poison_wrong_x.append(x)
            poison_wrong_w.append(w)
            poison_wrong_y.append(y)
    
    # Also separate wrong-classified bars
    benign_wrong_bars_x, benign_wrong_bars_w, benign_wrong_bars_y = [], [], []
    poison_wrong_bars_x, poison_wrong_bars_w, poison_wrong_bars_y = [], [], []
    for i, (x, w, y) in enumerate(zip(bx, bw, bwrong)):
        if y > 0:  # Only add if there are wrong classified
            benign_wrong_bars_x.append(x)
            benign_wrong_bars_w.append(w)
            benign_wrong_bars_y.append(y)
    for i, (x, w, y) in enumerate(zip(px, pw, pwrong)):
        if y > 0:  # Only add if there are wrong classified
            poison_wrong_bars_x.append(x)
            poison_wrong_bars_w.append(w)
            poison_wrong_bars_y.append(y)

    # Subplots: (1) histogram overlay + threshold, (2) boxplot (approx), (3) accuracy bars
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "<b>Score distribution</b>",
            "<b>Score comparison</b>",
            "<b>Accuracy by class</b>",
        ),
    )

    # (1) Histogram bars - render in order for proper layering
    # First: correct bars (behind)
    # Benign correct (below threshold)
    if benign_correct_x:
        fig.add_trace(
            go.Bar(
                x=benign_correct_x,
                y=benign_correct_y,
                width=benign_correct_w,
                name=f"<b>Benign (n={benign_total})</b>",
                marker=dict(
                    color="rgba(128, 128, 128, 0.85)",
                    line=dict(color="rgba(60, 60, 60, 1.0)", width=2),
                    pattern=dict(
                        shape=".",
                        fillmode="overlay",
                        size=4,
                        solidity=0.4,
                        fgcolor="rgba(60, 60, 60, 0.5)",
                    ),
                ),
                text=benign_correct_y,
                textposition="outside",
                textfont=dict(size=10, color="rgba(60, 60, 60, 1.0)", family="Times, serif"),
                opacity=0.9,
            ),
            row=1,
            col=1,
        )
    
    # Poison correct (above threshold)
    if poison_correct_x:
        fig.add_trace(
            go.Bar(
                x=poison_correct_x,
                y=poison_correct_y,
                width=poison_correct_w,
                name=f"<b>Poison (n={poison_total})</b>",
                marker=dict(
                    color="rgba(0, 180, 180, 0.85)",
                    line=dict(color="rgba(0, 140, 140, 1.0)", width=2),
                    pattern=dict(
                        shape="-",
                        fillmode="overlay",
                        size=5,
                        solidity=0.4,
                        fgcolor="rgba(0, 140, 140, 0.5)",
                    ),
                ),
                text=poison_correct_y,
                textposition="outside",
                textfont=dict(size=10, color="rgba(0, 140, 140, 1.0)", family="Times, serif"),
                opacity=0.9,
            ),
            row=1,
            col=1,
        )
    
    # Wrong-classified overlay bars (behind main bars)
    if benign_wrong_bars_x:
        fig.add_trace(
            go.Bar(
                x=benign_wrong_bars_x,
                y=benign_wrong_bars_y,
                width=benign_wrong_bars_w,
                name=f"<b>Benign wrong (n={benign_wrong})</b>",
                marker=dict(
                    color="rgba(60, 60, 60, 0.90)",
                    line=dict(color="rgba(40, 40, 40, 0.9)", width=1.5),
                ),
                opacity=0.85,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if poison_wrong_bars_x:
        fig.add_trace(
            go.Bar(
                x=poison_wrong_bars_x,
                y=poison_wrong_bars_y,
                width=poison_wrong_bars_w,
                name=f"<b>Poison wrong (n={poison_wrong})</b>",
                marker=dict(
                    color="rgba(0, 120, 120, 0.85)",
                    line=dict(color="rgba(0, 100, 100, 1.0)", width=2),
                ),
                opacity=0.9,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    
    # Then: wrong bars (on top) - Benign above threshold, Poison below threshold
    if benign_wrong_x:
        fig.add_trace(
            go.Bar(
                x=benign_wrong_x,
                y=benign_wrong_y,
                width=benign_wrong_w,
                name="",  # Don't duplicate in legend
                marker=dict(
                    color="rgba(128, 128, 128, 0.85)",
                    line=dict(color="rgba(60, 60, 60, 1.0)", width=2),
                    pattern=dict(
                        shape=".",
                        fillmode="overlay",
                        size=4,
                        solidity=0.4,
                        fgcolor="rgba(60, 60, 60, 0.5)",
                    ),
                ),
                text=benign_wrong_y,
                textposition="outside",
                textfont=dict(size=10, color="rgba(60, 60, 60, 1.0)", family="Times, serif"),
                opacity=0.9,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    
    if poison_wrong_x:
        fig.add_trace(
            go.Bar(
                x=poison_wrong_x,
                y=poison_wrong_y,
                width=poison_wrong_w,
                name="",  # Don't duplicate in legend
                marker=dict(
                    color="rgba(0, 180, 180, 0.85)",
                    line=dict(color="rgba(0, 140, 140, 1.0)", width=2),
                    pattern=dict(
                        shape="-",
                        fillmode="overlay",
                        size=5,
                        solidity=0.4,
                        fgcolor="rgba(0, 140, 140, 0.5)",
                    ),
                ),
                text=poison_wrong_y,
                textposition="outside",
                textfont=dict(size=10, color="rgba(0, 140, 140, 1.0)", family="Times, serif"),
                opacity=0.9,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Threshold line with text annotation and legend entry
    max_y = max(max(bcounts) if bcounts else [0], max(pcounts) if pcounts else [0])
    # Add invisible scatter with square marker for legend
    fig.add_trace(
        go.Scatter(
            x=[x_range[0] - (x_range[1] - x_range[0]) * 0.2],
            y=[-max_y * 0.1],
            mode="markers",
            name=f"<b>Threshold: {threshold:.4f}</b>",
            marker=dict(
                symbol="square",
                size=12,
                color="green",
                line=dict(color="green", width=1.5),
            ),
            showlegend=True,
            hoverinfo="skip",
            legendgroup="threshold",
        ),
        row=1,
        col=1,
    )
    # Add the actual line
    fig.add_trace(
        go.Scatter(
            x=[threshold, threshold],
            y=[0, max_y * 1.1],
            mode="lines",
            name="",
            line=dict(color="green", width=2.5, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
            legendgroup="threshold",
        ),
        row=1,
        col=1,
    )

    # Update X axis with range
    fig.update_xaxes(
        title_text="Anomaly Score",
        title_font=dict(size=13, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
        title_standoff=5,
        range=x_range,
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.08)",
        gridwidth=1,
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Frequency",
        title_font=dict(size=13, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.08)",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="rgba(0, 0, 0, 1.0)",
        zerolinewidth=2.5,
        range=[0, max_y * 1.1],
        tickfont=dict(size=11, family="Times, serif"),
        row=1,
        col=1,
    )

    # (2) Boxplot (approx) from expanded midpoints
    benign_vals = _expand_midpoints(benign_bins)
    poison_vals = _expand_midpoints(poison_bins)
    fig.add_trace(
        go.Box(
            y=benign_vals,
            name="Benign",
            marker_color="rgba(128, 128, 128, 0.75)",
            line_color="rgba(60, 60, 60, 0.9)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Box(
            y=poison_vals,
            name="Poison",
            marker_color="rgba(0, 180, 180, 0.75)",
            line_color="rgba(0, 140, 140, 0.9)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="green",
        line_width=2.5,
        row=1,
        col=2,
    )
    # Add text annotation in the middle of the threshold line (second subplot)
    # Calculate middle X position (boxplot has categorical x: 0 for Benign, 1 for Poison)
    fig.add_annotation(
        xref="x2",
        yref="y2",
        x=0.5,  # Middle between the two boxes
        y=threshold,
        text=f"{threshold:.4f}",
        showarrow=False,
        bgcolor="rgba(255, 250, 240, 0.9)",
        bordercolor="green",
        borderwidth=1.5,
        borderpad=3,
        font=dict(size=11, family="Times, serif", color="green"),
        align="center",
    )
    fig.update_xaxes(
        title_text="",
        title_standoff=5,
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.08)",
        tickfont=dict(size=13, family="Times, serif"),
        ticklabelstandoff=15,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="Anomaly Score",
        title_font=dict(size=13, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.08)",
        tickfont=dict(size=11, family="Times, serif"),
        row=1,
        col=2,
    )

    # (3) Accuracy bars
    fig.add_trace(
        go.Bar(
            x=["Benign", "Poison"],
            y=[benign_acc * 100.0, poison_acc * 100.0],
            marker=dict(
                color=["rgba(128, 128, 128, 0.75)", "rgba(0, 180, 180, 0.75)"],
                line=dict(color=["rgba(60, 60, 60, 0.9)", "rgba(0, 140, 140, 0.9)"], width=1.5),
            ),
            name="<b>Accuracy</b>",
            text=[f"{benign_acc*100:.1f}%", f"{poison_acc*100:.1f}%"],
            textposition="auto",
            textfont=dict(size=10, family="Times, serif"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="gray",
        line_width=1.5,
        row=1,
        col=3,
    )
    # Add text annotation in the middle of the 50% line (third subplot)
    # The x axis has two categories: "Benign" (0) and "Poison" (1), so middle is 0.5
    fig.add_annotation(
        xref="x3",
        yref="y3",
        x=0.5,  # Middle between Benign and Poison
        y=50,
        text="50%",
        showarrow=False,
        bgcolor="rgba(255, 250, 240, 0.9)",
        bordercolor="gray",
        borderwidth=1.5,
        borderpad=3,
        font=dict(size=11, family="Times, serif", color="gray"),
        align="center",
    )
    fig.update_xaxes(
        title_text="",
        title_standoff=5,
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.08)",
        tickfont=dict(size=13, family="Times, serif"),
        ticklabelstandoff=15,
        row=1,
        col=3,
    )
    fig.update_yaxes(
        title_text="Accuracy (%)",
        title_font=dict(size=13, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
        range=[0, 105],
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.08)",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="rgba(0, 0, 0, 1.0)",
        zerolinewidth=2.5,
        tickfont=dict(size=11, family="Times, serif"),
        row=1,
        col=3,
    )

    fig.update_layout(
        title=dict(
            text="<b>Evaluation Phase: Performance Assessment with Calibrated Threshold</b>",
            font=dict(size=15, family="Times, serif", color="rgba(0, 0, 0, 0.95)"),
            x=0.5,
            xanchor="center",
            pad=dict(b=5, t=5),
        ),
        barmode="overlay",
        bargap=0.05,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.02,  # Positioned at the left of the first subplot
            bgcolor="rgba(255, 250, 240, 0.85)",
            bordercolor="rgba(0, 0, 0, 0.25)",
            borderwidth=1,
            font=dict(size=12, family="Times, serif"),
            itemsizing="constant",
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
        plot_bgcolor="rgba(255, 250, 240, 1)",
        paper_bgcolor="white",
        margin=dict(l=50, r=35, t=50, b=40),
        hovermode="x unified",
        width=1500,
        height=450,
        autosize=False,
        font=dict(family="Times, serif"),
    )
    # Update subplot title fonts
    fig.update_annotations(font=dict(size=12, family="Times, serif"))

    fig.show()


if __name__ == "__main__":
    main()


