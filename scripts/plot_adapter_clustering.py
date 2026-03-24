#!/usr/bin/env python3
"""Build a clustering panel for benign vs poisoned LoRA adapters."""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.patches import Patch
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import ConnectionPatch
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_MODELS = ("gemma", "llama", "qwen")


COLORS = {
    "benign": "#25D366",
    "poison": "#F03E3E",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a clean 2D clustering panel for benign vs poisoned adapters."
    )
    parser.add_argument(
        "--mode",
        default="schematic",
        choices=("schematic", "data"),
        help=(
            "Plotting mode. 'schematic' generates a clean illustrative 2D panel; "
            "'data' uses real adapter features."
        ),
    )
    parser.add_argument(
        "--model",
        default="qwen",
        choices=DEFAULT_MODELS,
        help="Model family used to resolve directories in data mode.",
    )
    parser.add_argument(
        "--benign-dir",
        help="Directory with benign adapters. Defaults to output_<model>/benign.",
    )
    parser.add_argument(
        "--poison-dir",
        help="Directory with poisoned adapters. Defaults to output_<model>/poison.",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=20,
        help="Layer index used for feature extraction.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help="Optional cap on the number of adapters per class.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for class-balanced sampling.",
    )
    parser.add_argument(
        "--feature-mode",
        default="projection-wise",
        choices=("five-metric", "projection-wise"),
        help=(
            "Feature representation used for clustering. "
            "'projection-wise' keeps the full 20D detector representation; "
            "'five-metric' averages the same metric across q/k/v/o and yields "
            "a compact 5D representation."
        ),
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help=(
            "Output path without extension. Defaults to "
            "iclr2026/images/adapter_clustering_<model>."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title. If omitted, no title is added.",
    )
    return parser.parse_args()


def list_adapter_dirs(directory: Path, expected_type: str):
    if not directory.exists():
        return []

    paths = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            continue
        if metadata.get("type") == expected_type:
            paths.append(entry)
    return paths


def sample_paths(paths, limit, rng):
    if limit is None or len(paths) <= limit:
        return paths
    indices = np.sort(rng.choice(len(paths), size=limit, replace=False))
    return [paths[i] for i in indices]


def set_publication_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def make_schematic_clusters(random_seed):
    rng = np.random.default_rng(random_seed)

    benign_main = rng.multivariate_normal(
        mean=[-1.55, 0.1],
        cov=[[0.13, 0.04], [0.04, 0.09]],
        size=55,
    )
    benign_tail = rng.multivariate_normal(
        mean=[-0.8, -0.45],
        cov=[[0.03, 0.0], [0.0, 0.025]],
        size=10,
    )
    poison_main = rng.multivariate_normal(
        mean=[1.6, -0.05],
        cov=[[0.12, -0.03], [-0.03, 0.08]],
        size=52,
    )
    poison_tail = rng.multivariate_normal(
        mean=[0.95, 0.45],
        cov=[[0.028, 0.0], [0.0, 0.022]],
        size=10,
    )

    benign = np.vstack([benign_main, benign_tail])
    poison = np.vstack([poison_main, poison_tail])
    return benign, poison


def make_schematic_clusters_3d(random_seed):
    rng = np.random.default_rng(random_seed)
    benign = rng.multivariate_normal(
        mean=[1.7, 3.15, 1.45],
        cov=[
            [0.24, 0.06, 0.05],
            [0.06, 0.34, 0.07],
            [0.05, 0.07, 0.26],
        ],
        size=60,
    )
    benign[:, 0] += rng.normal(0.0, 0.16, size=len(benign))
    benign[:, 0] *= 2.0
    benign[:, 2] *= 0.5

    poison = rng.multivariate_normal(
        mean=[3.85, 2.55, 2.45],
        cov=[
            [0.28, -0.05, 0.07],
            [-0.05, 0.22, 0.05],
            [0.07, 0.05, 0.24],
        ],
        size=60,
    )
    poison[:, 1] += rng.normal(0.0, 0.18, size=len(poison))
    poison[:, 0] *= 2.0
    return benign, poison


def project_points_3d(points):
    basis = np.array(
        [
            [-0.92, -0.34],  # 0 dim: front-facing axis, tilted leftward
            [1.06, -0.62],   # 1 dim: strongly tilted down-right
            [0.0, 1.0],     # 2 dim
        ],
        dtype=np.float64,
    )
    return points @ basis


def get_perspective_axes():
    return {
        "0 dim": np.array([-3.9, -1.25]),
        "1 dim": np.array([3.27, -1.87]),
        "2 dim": np.array([0.0, 3.8]),
    }


def extract_feature_matrix(paths, layer_idx):
    from core.detector import BackdoorDetector

    features = []
    kept_paths = []
    for path in paths:
        feat = BackdoorDetector._extract_features_from_adapter(path, layer_idx)
        if feat is None:
            continue
        features.append(feat)
        kept_paths.append(path)
    if not features:
        return np.empty((0, 20), dtype=np.float32), kept_paths
    return np.vstack(features), kept_paths


def project_feature_space(X, feature_mode):
    if feature_mode == "projection-wise":
        return X
    if X.shape[1] != 20:
        raise ValueError(f"Expected 20 features for projection reduction, got {X.shape[1]}")
    return X.reshape(len(X), 4, 5).mean(axis=1)


def add_cluster_hull(ax, points, color):
    if len(points) < 3:
        return

    try:
        hull = ConvexHull(points)
    except Exception:
        return

    hull_points = points[hull.vertices]
    if len(hull_points) < 3:
        return

    closed = np.vstack([hull_points, hull_points[0]])

    if len(hull_points) >= 4:
        try:
            tck, _ = splprep([closed[:, 0], closed[:, 1]], s=0.4, per=True)
            u_new = np.linspace(0, 1, 220)
            x_new, y_new = splev(u_new, tck)
            contour_points = np.column_stack([x_new, y_new])
        except Exception:
            contour_points = closed
    else:
        contour_points = closed

    patch = Polygon(
        contour_points,
        closed=True,
        facecolor=color,
        edgecolor=color,
        alpha=0.14,
        linewidth=2.0,
        zorder=1,
        joinstyle="round",
    )
    ax.add_patch(patch)


def add_density_cloud(ax, points, color, levels=10):
    if len(points) < 6:
        return

    x = points[:, 0]
    y = points[:, 1]
    pad_x = max(np.ptp(x) * 0.42, 0.45)
    pad_y = max(np.ptp(y) * 0.42, 0.45)

    xx, yy = np.mgrid[
        (x.min() - pad_x):(x.max() + pad_x):220j,
        (y.min() - pad_y):(y.max() + pad_y):220j,
    ]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])

    kde = gaussian_kde(values, bw_method=0.5)
    density = np.reshape(kde(positions), xx.shape)
    density = density / max(density.max(), 1e-9)

    cmap = LinearSegmentedColormap.from_list(
        f"cloud_{color.replace('#', '')}",
        [
            (1.0, 1.0, 1.0, 0.0),
            (*plt.matplotlib.colors.to_rgb(color), 0.08),
            (*plt.matplotlib.colors.to_rgb(color), 0.18),
            (*plt.matplotlib.colors.to_rgb(color), 0.30),
            (*plt.matplotlib.colors.to_rgb(color), 0.48),
            (*plt.matplotlib.colors.to_rgb(color), 0.68),
            (*plt.matplotlib.colors.to_rgb(color), 0.90),
        ],
    )

    level_values = np.linspace(0.02, 1.0, 36)
    ax.contourf(
        xx,
        yy,
        density,
        levels=level_values,
        cmap=cmap,
        antialiased=True,
        zorder=2,
    )


def region_anchor(points, x_shift=0.0, y_shift=0.0):
    center = points.mean(axis=0)
    return center[0] + x_shift, center[1] + y_shift


def add_separator_curve(ax, benign_points, poison_points):
    benign_center = benign_points.mean(axis=0)
    poison_center = poison_points.mean(axis=0)
    midpoint = 0.5 * (benign_center + poison_center)

    dx = poison_center[0] - benign_center[0]
    dy = poison_center[1] - benign_center[1]
    norm = max(np.hypot(dx, dy), 1e-6)

    # Normal direction to the line connecting the two cluster centers.
    nx = -dy / norm
    ny = dx / norm

    span = max(
        np.ptp(np.concatenate([benign_points[:, 1], poison_points[:, 1]])),
        np.ptp(np.concatenate([benign_points[:, 0], poison_points[:, 0]])) * 0.45,
        2.6,
    )
    arc = max(norm * 0.18, 0.45)

    t = np.linspace(-1.0, 1.0, 240)
    base_x = midpoint[0] + t * span * nx
    base_y = midpoint[1] + t * span * ny

    # Bend the separator towards a smooth S-like curve instead of a straight line.
    curve = arc * np.sin(np.pi * t)
    x = base_x + curve * dx / norm
    y = base_y + curve * dy / norm

    ax.plot(
        x,
        y,
        color="#5E6472",
        linewidth=2.2,
        linestyle=(0, (6, 5)),
        alpha=0.9,
        zorder=2,
    )


def draw_perspective_axes(ax, axes=None):
    origin = np.array([0.0, 0.0])
    if axes is None:
        axes = get_perspective_axes()

    for label, endpoint in axes.items():
        ax.annotate(
            "",
            xy=endpoint,
            xytext=origin,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.45,
                color="#7A8190",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=12,
            ),
            zorder=1,
        )
        if label == "0 dim":
            label_pos = endpoint + np.array([-0.50, 0.18])
        elif label == "1 dim":
            label_pos = endpoint + np.array([0.08, -0.06])
        else:
            label_pos = endpoint + np.array([0.08, 0.08])
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            fontsize=10.5,
            color="#4A5568",
            ha="left",
            va="bottom",
        )

    ax.scatter(
        [origin[0]],
        [origin[1]],
        s=20,
        c="#4A5568",
        zorder=4,
    )


def add_box(
    ax,
    xy,
    width,
    height,
    text,
    fc="#FFFFFF",
    ec="#CBD5E0",
    lw=1.2,
    fs=9.5,
    radius=0.02,
    pad=0.0015,
):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#1A202C",
        zorder=3,
    )
    return box


def draw_feature_pipeline(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    adapter_box = add_box(ax, (0.04, 0.45), 0.20, 0.10, "LoRA Adapter", fc="#F8FAFC", ec="#94A3B8", fs=10)

    proj_x = 0.31
    metric_x = 0.47
    proj_y = [0.73, 0.56, 0.39, 0.22]
    proj_labels = [r"$q_{\mathrm{proj}}$", r"$k_{\mathrm{proj}}$", r"$v_{\mathrm{proj}}$", r"$o_{\mathrm{proj}}$"]
    metric_boxes = []

    ax.text(
        0.55,
        0.92,
        "projection-wise spectral features",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#111827",
    )

    adapter_mid_y = adapter_box.get_y() + adapter_box.get_height() / 2
    stack_mid_y = np.mean([y + 0.0325 for y in proj_y])
    ax.annotate(
        "",
        xy=(proj_x - 0.03, stack_mid_y),
        xytext=(adapter_box.get_x() + adapter_box.get_width(), adapter_mid_y),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#111827", mutation_scale=7),
        zorder=1,
    )

    for y, label in zip(proj_y, proj_labels):
        proj_box = add_box(ax, (proj_x, y), 0.11, 0.065, label, fc="#EEF2FF", ec="#A5B4FC", fs=10.5)
        metric_box = add_box(
            ax,
            (metric_x, y - 0.05),
            0.14,
            0.17,
            "$\\sigma_1$\n$\\|\\Delta W\\|_F$\n$E_{\\sigma_1}$\n$H$\n$K$",
            fc="#FFFFFF",
            ec="#CBD5E0",
            fs=7.3,
            radius=0.018,
        )
        metric_boxes.append(metric_box)
        ax.annotate(
            "",
            xy=(metric_x, y + 0.0325),
            xytext=(proj_x + 0.11, y + 0.0325),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#111827", mutation_scale=7),
            zorder=1,
        )

    concat_box = add_box(ax, (0.69, 0.47), 0.12, 0.06, "concat", fc="#F8FAFC", ec="#94A3B8", fs=9)
    vector_box = add_box(
        ax,
        (0.84, 0.44),
        0.15,
        0.10,
        r"$\vec{\mathbf{v}} \in \mathbb{R}^{20}$",
        fc="#FFF7ED",
        ec="#FB923C",
        fs=10,
    )

    for box in metric_boxes:
        y_mid = box.get_y() + box.get_height() / 2
        ax.annotate(
            "",
            xy=(0.69, 0.50),
            xytext=(box.get_x() + box.get_width(), y_mid),
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#111827", mutation_scale=7),
            zorder=1,
        )

    ax.annotate(
        "",
        xy=(0.84, 0.49),
        xytext=(0.81, 0.50),
        arrowprops=dict(arrowstyle="-|>", lw=1.05, color="#111827", mutation_scale=7),
        zorder=1,
    )
    vector_anchor = (vector_box.get_x() + vector_box.get_width(), vector_box.get_y() + vector_box.get_height() / 2)
    return vector_anchor

def save_multi_format(fig, output_stem: Path):
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    svg_path = output_stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def save_axes_crops(fig, output_stem: Path, axes_map: dict[str, any]):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for suffix, ax in axes_map.items():
        bbox = ax.get_tightbbox(renderer).expanded(1.02, 1.05)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        stem = output_stem.with_name(f"{output_stem.name}_{suffix}")
        stem.parent.mkdir(parents=True, exist_ok=True)
        png_path = stem.with_suffix(".png")
        svg_path = stem.with_suffix(".svg")
        fig.savefig(png_path, dpi=300, bbox_inches=bbox_inches)
        fig.savefig(svg_path, bbox_inches=bbox_inches)
        print(f"Saved {png_path}")
        print(f"Saved {svg_path}")


def build_schematic_left_panel(output_stem: Path, title=None):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    draw_feature_pipeline(ax)
    if title:
        fig.suptitle(title, fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.08)
    save_multi_format(fig, output_stem)
    plt.close(fig)


def build_plot_from_points(
    benign_points,
    poison_points,
    output_stem: Path,
    title=None,
    draw_hulls=False,
    draw_separator=False,
    draw_axes=True,
):
    set_publication_style()

    fig, ax = plt.subplots(figsize=(6.4, 4.9))

    for points, class_name, label in [
        (benign_points, "benign", "Benign"),
        (poison_points, "poison", "Poisoned"),
    ]:
        color = COLORS[class_name]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=56,
            alpha=0.95,
            facecolors=color,
            edgecolors=color,
            linewidths=1.3,
            zorder=3,
            label=label,
        )
        ax.collections[-1].set_alpha(0.18)
        if draw_hulls:
            add_cluster_hull(ax, points, color=color)

    if draw_separator:
        add_separator_curve(ax, benign_points, poison_points)

    bx, by = region_anchor(benign_points, x_shift=-0.08, y_shift=0.18)
    px, py = region_anchor(poison_points, x_shift=0.08, y_shift=-0.18)

    ax.text(
        bx,
        by,
        "Benign cluster",
        color=COLORS["benign"],
        fontsize=10,
        fontweight="semibold",
        ha="center",
        va="center",
        zorder=4,
    )
    ax.text(
        px,
        py,
        "Poisoned cluster",
        color=COLORS["poison"],
        fontsize=10,
        fontweight="semibold",
        ha="center",
        va="center",
        zorder=4,
    )

    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.12, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(draw_axes)
    ax.spines["left"].set_visible(draw_axes)
    if draw_axes:
        ax.set_xlabel("0 dim")
        ax.set_ylabel("1 dim")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["benign"],
               markeredgecolor=COLORS["benign"], markeredgewidth=1.3, markersize=8, label="Benign"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["poison"],
               markeredgecolor=COLORS["poison"], markeredgewidth=1.3, markersize=8, label="Poisoned"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", handletextpad=0.5, borderpad=0.2)
    if draw_axes:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    save_multi_format(fig, output_stem)
    plt.close(fig)


def build_schematic_right_panel(output_stem: Path, random_seed: int, title=None):
    set_publication_style()
    benign_3d, poison_3d = make_schematic_clusters_3d(random_seed)
    benign_points = project_points_3d(benign_3d)
    poison_points = project_points_3d(poison_3d)

    axes = get_perspective_axes()
    z_axis = axes["2 dim"]
    x_axis = axes["0 dim"]
    y_axis = axes["1 dim"]

    z_unit = z_axis / max(np.linalg.norm(z_axis), 1e-6)
    x_unit = x_axis / max(np.linalg.norm(x_axis), 1e-6)
    y_unit = y_axis / max(np.linalg.norm(y_axis), 1e-6)
    # Bias the in-plane direction toward the x-axis while keeping a visible y component.
    xy_direction = 0.72 * x_unit + 0.28 * y_unit
    xy_bisector = xy_direction / max(np.linalg.norm(xy_direction), 1e-6)

    angle_deg = 30.0
    angle_rad = np.deg2rad(angle_deg)
    length = 3.55
    adapter_point = (
        np.cos(angle_rad) * length * z_unit
        + np.sin(angle_rad) * length * xy_bisector
    )

    # Enlarge the clouds roughly 4x in area (2x radial scale) and align their
    # vertical centers with the adapter-vector endpoint.
    def scale_and_lift(points, target_y):
        center = points.mean(axis=0)
        scaled = center + 2.0 * (points - center)
        scaled[:, 1] += target_y - scaled[:, 1].mean()
        return scaled

    poison_target_y = adapter_point[1]
    benign_target_y = adapter_point[1] * (0.5 / 1.3)

    benign_points = scale_and_lift(benign_points, benign_target_y)
    poison_points = scale_and_lift(poison_points, poison_target_y)

    # Pull the clouds and adapter representation leftward so the two panels read as one figure,
    # but keep the coordinate axes fixed to preserve the intended geometry.
    scene_shift_x = -1.05
    benign_points[:, 0] += scene_shift_x + 0.5
    benign_points[:, 1] -= 0.5
    poison_points[:, 0] += scene_shift_x + 2.5
    adapter_point[0] += scene_shift_x

    # Make the red cloud slightly tighter only on the left side of its own center.
    poison_center_x = poison_points[:, 0].mean()
    left_mask = poison_points[:, 0] < poison_center_x
    poison_points[left_mask, 0] = (
        poison_center_x + 0.72 * (poison_points[left_mask, 0] - poison_center_x)
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    draw_perspective_axes(ax, axes=axes)

    for points, class_name in [(benign_points, "benign"), (poison_points, "poison")]:
        color = COLORS[class_name]
        add_density_cloud(ax, points, color=color)

    ax.annotate(
        "",
        xy=adapter_point,
        xytext=(0.0, 0.0),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.45,
            color="#1A202C",
            mutation_scale=11,
        ),
        zorder=5,
    )
    ax.scatter(
        [adapter_point[0]],
        [adapter_point[1]],
        s=46,
        facecolors="white",
        edgecolors="#1A202C",
        linewidths=1.15,
        zorder=6,
    )
    ax.text(
        adapter_point[0] + 0.18,
        adapter_point[1] + 0.08,
        "Adapter",
        fontsize=10.5,
        color="#1A202C",
        ha="left",
        va="center",
        zorder=6,
    )

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)

    # Leave enough breathing room so the density clouds fade to white before clipping.
    ax.set_xlim(-5.75, 5.9)
    ax.set_ylim(-3.55, 4.85)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.55)
    save_multi_format(fig, output_stem)
    plt.close(fig)


def build_schematic_3d_plot(output_stem: Path, random_seed: int, title=None):
    build_schematic_left_panel(output_stem.with_name(f"{output_stem.name}_left"), title=None)
    build_schematic_right_panel(output_stem.with_name(f"{output_stem.name}_right"), random_seed=random_seed, title=title)


def build_plot(X, y, output_stem: Path, title=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_scaled)
    benign_points = Z[y == 0]
    poison_points = Z[y == 1]
    build_plot_from_points(
        benign_points,
        poison_points,
        output_stem=output_stem,
        title=title,
        draw_hulls=True,
        draw_separator=False,
    )


def main():
    args = parse_args()
    import config

    default_stem_name = (
        "adapter_clustering_panel"
        if args.mode == "schematic"
        else f"adapter_clustering_{args.model}"
    )
    output_stem = Path(
        args.output_stem
        or (Path(config.ROOT_DIR) / "iclr2026" / "images" / default_stem_name)
    )

    if args.mode == "schematic":
        build_schematic_3d_plot(
            output_stem=output_stem,
            random_seed=args.random_seed,
            title=args.title,
        )
        return

    benign_dir = Path(args.benign_dir or (Path(config.ROOT_DIR) / f"output_{args.model}/benign"))
    poison_dir = Path(args.poison_dir or (Path(config.ROOT_DIR) / f"output_{args.model}/poison"))

    rng = np.random.default_rng(args.random_seed)
    benign_paths = sample_paths(list_adapter_dirs(benign_dir, "benign"), args.sample_per_class, rng)
    poison_paths = sample_paths(list_adapter_dirs(poison_dir, "poison"), args.sample_per_class, rng)

    benign_X, benign_kept = extract_feature_matrix(benign_paths, args.layer_idx)
    poison_X, poison_kept = extract_feature_matrix(poison_paths, args.layer_idx)

    if len(benign_kept) == 0 or len(poison_kept) == 0:
        raise SystemExit("Could not extract features for one or both classes.")

    benign_repr = project_feature_space(benign_X, args.feature_mode)
    poison_repr = project_feature_space(poison_X, args.feature_mode)

    X = np.vstack([benign_repr, poison_repr])
    y = np.concatenate(
        [
            np.zeros(len(benign_kept), dtype=np.int32),
            np.ones(len(poison_kept), dtype=np.int32),
        ]
    )

    print(f"Benign adapters used: {len(benign_kept)}")
    print(f"Poison adapters used: {len(poison_kept)}")
    print(f"Feature shape: {X.shape}")
    print(f"Feature mode: {args.feature_mode}")

    build_plot(X, y, output_stem=output_stem, title=args.title)


if __name__ == "__main__":
    main()
