import plotly.graph_objects as go


# Paste your binned hist data here (bin_start, bin_end, count)
BENIGN_BINS = [
    (8.63, 11.65, 12),
    (11.65, 14.67, 2),
    (26.54, 29.56, 2),
    (29.56, 32.58, 1),
    (35.62, 38.64, 3),
    (38.64, 41.66, 6),
    (41.66, 44.68, 7),
    (44.68, 47.70, 5),
    (50.54, 53.56, 11),
    (53.56, 56.58, 6),
    (56.58, 59.60, 8),
    (59.60, 62.62, 7),
    (62.62, 65.64, 6),
    (65.64, 68.66, 4),
]

POISON_BINS = [
    (80.57, 81.14, 2),
    (82.35, 82.92, 2),
    (84.06, 84.66, 1),
    (85.14, 85.71, 1),
    (85.71, 86.28, 3),
    (86.28, 86.85, 1),
    (86.85, 87.42, 1),
    (87.42, 87.99, 1),
    (88.70, 89.27, 4),
    (89.27, 89.84, 3),
    (91.55, 92.12, 1),
]


# Threshold (use 71.79 if your axis is 0–100; use 0.7179 if your axis is 0–1)
THRESHOLD = 71.79


def _maybe_rescale_to_unit_interval(bins: list[tuple[float, float, int]], threshold: float):
    """
    If values look like 0–100, rescale to 0–1.
    Heuristic: if any edge > 1.5, treat as percent-like.
    """
    max_edge = max(b for a, b, _ in bins)
    if max_edge > 1.5:
        scaled_bins = [(a / 100.0, b / 100.0, c) for a, b, c in bins]
        return scaled_bins, threshold / 100.0, True
    return bins, threshold, False


def _to_bar_arrays(bins: list[tuple[float, float, int]], min_width: float = None, prevent_overlap: bool = False, x_range: tuple = None):
    centers = []
    widths = []
    counts = []
    
    if prevent_overlap and len(bins) > 0:
        # Sort bins by start position
        sorted_bins = sorted(bins, key=lambda x: x[0])
        adjusted_data = {}  # bin -> (center, width, count)
        
        for i, (a, b, c) in enumerate(sorted_bins):
            raw_width = max(b - a, 0.0)
            if min_width is not None:
                width = max(raw_width, min_width)
            else:
                width = raw_width
            
            # Calculate center
            center = (a + b) / 2.0
            
            # Check for overlap with previous bar
            if i > 0:
                prev_bin = sorted_bins[i-1]
                prev_center, prev_width, _ = adjusted_data[prev_bin]
                prev_end = prev_center + prev_width / 2.0
                current_start = center - width / 2.0
                
                # If overlap, adjust position
                if current_start < prev_end:
                    # Move current bar to start right after previous with small gap
                    if x_range:
                        gap = (x_range[1] - x_range[0]) * 0.001  # Small gap
                    else:
                        # Calculate from bins range
                        all_edges = [edge for bin_val in bins for edge in (bin_val[0], bin_val[1])]
                        gap = (max(all_edges) - min(all_edges)) * 0.001
                    center = prev_end + width / 2.0 + gap
            
            adjusted_data[(a, b, c)] = (center, width, c)
        
        # Reorder to match original order
        for bin_val in bins:
            center, width, count = adjusted_data[bin_val]
            centers.append(center)
            widths.append(width)
            counts.append(count)
    else:
        for a, b, c in bins:
            centers.append((a + b) / 2.0)
            raw_width = max(b - a, 0.0)
            # Apply minimum width if provided
            if min_width is not None:
                widths.append(max(raw_width, min_width))
            else:
                widths.append(raw_width)
            counts.append(c)
    
    return centers, widths, counts


def main():
    benign_bins, threshold, scaled = _maybe_rescale_to_unit_interval(BENIGN_BINS, THRESHOLD)
    poison_bins, _, _ = _maybe_rescale_to_unit_interval(POISON_BINS, THRESHOLD)

    # Calculate minimum width based on axis range
    x_range = [0, 1] if scaled else [0, 100]
    min_width_benign = (x_range[1] - x_range[0]) * 0.02  # 2% of total range for better visibility
    
    # For poison, calculate min width that respects gaps between bins
    # Find minimum gap between consecutive bins and minimum bin width
    sorted_poison = sorted(poison_bins, key=lambda x: x[0])
    min_gap = float('inf')
    min_bin_width = float('inf')
    
    for i in range(len(sorted_poison) - 1):
        gap = sorted_poison[i+1][0] - sorted_poison[i][1]
        if gap > 0:
            min_gap = min(min_gap, gap)
        # Also track minimum bin width
        bin_width = sorted_poison[i][1] - sorted_poison[i][0]
        min_bin_width = min(min_bin_width, bin_width)
    
    # Check last bin width
    if len(sorted_poison) > 0:
        last_bin_width = sorted_poison[-1][1] - sorted_poison[-1][0]
        min_bin_width = min(min_bin_width, last_bin_width)
    
    # Use smaller min_width for poison: don't exceed actual bin widths or gaps
    if min_gap != float('inf') and min_bin_width != float('inf'):
        # Use the smaller of: 1% of range, 80% of min gap, or actual min bin width
        min_width_poison = min((x_range[1] - x_range[0]) * 0.01, min_gap * 0.8, min_bin_width)
    elif min_bin_width != float('inf'):
        min_width_poison = min((x_range[1] - x_range[0]) * 0.01, min_bin_width)
    else:
        min_width_poison = (x_range[1] - x_range[0]) * 0.01

    bx, bw, by = _to_bar_arrays(benign_bins, min_width=min_width_benign, prevent_overlap=False, x_range=x_range)
    px, pw, py = _to_bar_arrays(poison_bins, min_width=min_width_poison, prevent_overlap=False, x_range=x_range)

    total_benign = sum(by)
    total_poison = sum(py)
    
    # Calculate max frequency for Y axis range
    max_frequency = max(max(by) if by else [0], max(py) if py else [0])

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bx,
            y=by,
            width=bw,
            name=f"<b>Benign (n={total_benign})</b>",
            marker=dict(
                color="rgba(128, 128, 128, 0.75)",
                line=dict(color="rgba(60, 60, 60, 0.9)", width=1.5),
                pattern=dict(
                    shape=".",
                    fillmode="overlay",
                    size=4,
                    solidity=0.3,
                    fgcolor="rgba(60, 60, 60, 0.4)",
                ),
            ),
            text=by,
            textposition="outside",
            textfont=dict(size=10, color="rgba(60, 60, 60, 1.0)", family="Times, serif"),
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Bar(
            x=px,
            y=py,
            width=pw,
            name=f"<b>Poison (n={total_poison})</b>",
            marker=dict(
                color="rgba(0, 180, 180, 0.75)",
                line=dict(color="rgba(0, 140, 140, 0.9)", width=1.5),
                pattern=dict(
                    shape="-",
                    fillmode="overlay",
                    size=5,
                    solidity=0.3,
                    fgcolor="rgba(0, 140, 140, 0.4)",
                ),
            ),
            text=py,
            textposition="outside",
            textfont=dict(size=10, color="rgba(0, 140, 140, 1.0)", family="Times, serif"),
            opacity=0.85,
        )
    )

    # X axis range: show full 0–1 if scaled, otherwise 0–100 (so you can compare visually)
    # x_range already calculated above

    # Threshold line - add as trace to appear in legend with square marker
    max_y = max(max(by) if by else [0], max(py) if py else [0])
    # Add invisible scatter with square marker for legend (square symbol like bars)
    fig.add_trace(
        go.Scatter(
            x=[x_range[0] - (x_range[1] - x_range[0]) * 0.2],  # Position outside visible area
            y=[-max_y * 0.1],  # Position below visible area
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
        )
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
        )
    )

    # Also add the vline for visual reference
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="green",
        line_width=2.5,
        annotation_text="",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(
            text="<b>Calibration Phase: Anomaly Score Distribution for Threshold Determination</b>",
            font=dict(size=15, family="Times, serif", color="rgba(0, 0, 0, 0.95)"),
            x=0.5,
            xanchor="center",
            pad=dict(b=5, t=5),
        ),
        xaxis=dict(
            title=dict(
                text="Anomaly Score",
                font=dict(size=13, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
                standoff=5,
            ),
            range=x_range,
            showgrid=True,
            gridcolor="rgba(0, 0, 0, 0.08)",
            gridwidth=1,
            zeroline=False,
            showline=False,
            tickfont=dict(size=11, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
        ),
        yaxis=dict(
            title=dict(
                text="Frequency",
                font=dict(size=13, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
                standoff=5,
            ),
            showgrid=True,
            gridcolor="rgba(0, 0, 0, 0.08)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="rgba(0, 0, 0, 1.0)",
            zerolinewidth=2.5,
            range=[0, max_frequency * 1.1],
            tickfont=dict(size=11, family="Times, serif"),
        ),
        barmode="overlay",
        bargap=0.05,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.98,
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
        width=1000,
        height=450,
        autosize=False,
    )

    fig.show()


if __name__ == "__main__":
    main()
