#!/usr/bin/env python3
"""
Detection efficacy vs finetuned layer and LoRA rank, with backdoor transfer estimate
as bubble size.

Reads merged_probe_panel.json.

Writes:
  backdoor_detection_correlation.png
  backdoor_detection_correlation.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ── palette identical to the appendix heatmap ─────────────────────────────────
_CMAP = LinearSegmentedColormap.from_list(
    "appendix_pink_blue",
    [
        (0.00, "#99004D"),
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
_RANK_T = {8: 0.06, 16: 0.50, 32: 0.94}


def _rank_color(rank: int) -> str:
    return mpl.colors.to_hex(_CMAP(_RANK_T.get(rank, 0.5)))


mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT_DIR = ROOT / "resultsFinal" / "layerRankPlots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = ROOT / "plotScripts" / "layerRankPlots" / "merged_probe_panel.json"
OUT_PNG = OUT_DIR / "backdoor_detection_correlation.png"
OUT_JSON = OUT_DIR / "backdoor_detection_correlation.json"

_W_MAX, _W_MEAN = 0.6, 0.4      # transfer estimate weights


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        from scipy import stats
        r, p = stats.spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        c = np.corrcoef(rx, ry)[0, 1]
        return float(c) if np.isfinite(c) else float("nan"), float("nan")


def main() -> None:
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    tags: list[str] = list(data.get("tags") or list(data.get("summary_at_lora_layer", {}).keys()))
    configs = data["configs"]
    summary = data.get("summary_at_lora_layer") or {}

    rows: list[dict] = []
    for tag in tags:
        if tag not in configs or tag not in summary:
            continue
        cfg = configs[tag]
        li = int(cfg["layer_idx"])
        probe = np.asarray(cfg["probe_test_acc"], dtype=float)
        if probe.size <= li:
            continue
        mask = np.ones(probe.size, dtype=bool)
        mask[li] = False
        remote = probe[mask]
        remote = remote[np.isfinite(remote)]
        rmx = float(np.nanmax(remote)) if remote.size else float("nan")
        rmean = float(np.nanmean(remote)) if remote.size else float("nan")
        bte = _W_MAX * rmx + _W_MEAN * rmean if (np.isfinite(rmx) and np.isfinite(rmean)) else float("nan")
        at = summary[tag]["at_lora_layer"]
        rows.append({
            "tag": tag,
            "rank": int(cfg["rank"]),
            "paper_layer": int(cfg["paper_layer"]),
            "layer_idx": li,
            "detection_roc_auc": float(at["roc_auc_test"]),
            "local_probe_test_acc": float(at["probe_test_acc"]),
            "local_kl_symmetric": float(at["kl_symmetric"]),
            "remote_probe_max": rmx,
            "remote_probe_mean": rmean,
            "backdoor_transfer_estimate": float(bte),
        })

    if len(rows) < 3:
        raise SystemExit(f"Need >=3 configs; found {len(rows)}")

    # Correlation stats for JSON + annotation
    pl = np.array([r["paper_layer"] for r in rows], dtype=float)
    rnk = np.array([r["rank"] for r in rows], dtype=float)
    det = np.array([r["detection_roc_auc"] for r in rows])
    te = np.array([r["backdoor_transfer_estimate"] for r in rows])
    rho_layer, p_layer = _spearman(pl, det)
    rho_rank, p_rank = _spearman(rnk, det)
    rho_te, p_te = _spearman(te, det)

    out_doc = {
        "n_configs": len(rows),
        "transfer_estimate_formula": f"{_W_MAX}×max_j≠LoRA(probe_acc[j]) + {_W_MEAN}×mean",
        "correlations": {
            "detection_vs_paper_layer": {"spearman_r": rho_layer, "spearman_p": p_layer},
            "detection_vs_rank": {"spearman_r": rho_rank, "spearman_p": p_rank},
            "detection_vs_transfer_estimate": {"spearman_r": rho_te, "spearman_p": p_te},
        },
        "rows": rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2)

    # ── shared helpers ────────────────────────────────────────────────────────
    t_min, t_max = float(te.min()), float(te.max())
    _SZ = 260   # fixed marker size — uniform across both panels

    _dx: dict[str, float] = {r["tag"]: 0.0 for r in rows}
    rank_at_l21 = sorted([r for r in rows if r["paper_layer"] == 21], key=lambda r: r["rank"])
    if len(rank_at_l21) > 1:
        n = len(rank_at_l21)
        offsets = np.linspace(-0.65 * (n - 1) / 2, 0.65 * (n - 1) / 2, n)
        for r, off in zip(rank_at_l21, offsets):
            _dx[r["tag"]] = float(off)

    def _style_ax(a: plt.Axes) -> None:
        a.set_facecolor("white")
        layer_ticks = [3, 9, 15, 21, 27]
        a.set_xticks(layer_ticks)
        a.set_xticklabels([f"L{l}" for l in layer_ticks], fontsize=17)
        a.set_xlabel("Finetuned layer", fontsize=18, fontweight="bold", labelpad=10)
        a.tick_params(axis="both", labelsize=16, width=1.5, length=6, pad=5)
        a.set_xlim(0, 30.5)
        a.grid(True, linestyle=":", color="#CCCCCC", linewidth=1.0, alpha=0.9, zorder=0)
        a.set_axisbelow(True)
        for sp in a.spines.values():
            sp.set_linewidth(1.5)
            sp.set_edgecolor("#222222")

    def _scatter_panel(a: plt.Axes, y_key: str) -> None:
        r16 = sorted([r for r in rows if r["rank"] == 16], key=lambda r: r["paper_layer"])
        a.plot(
            [r["paper_layer"] + _dx[r["tag"]] for r in r16],
            [r[y_key] for r in r16],
            color=_rank_color(16), linewidth=1.8, linestyle="--", alpha=0.45, zorder=1,
        )
        for row in rows:
            xp = row["paper_layer"] + _dx[row["tag"]]
            a.scatter(xp, row[y_key], s=_SZ,
                      color=_rank_color(row["rank"]),
                      edgecolors="#111111", linewidths=1.8,
                      zorder=3, clip_on=False)

    # ── figure 1×2 ────────────────────────────────────────────────────────────
    fig, (ax_det, ax_tr) = plt.subplots(1, 2, figsize=(13.0, 6.2),
                                         constrained_layout=True)
    fig.suptitle(
        "Effect of LoRA Placement on Detection Efficacy and Backdoor Transfer",
        fontsize=23, fontweight="bold", y=1.07,
    )

    # left — detection
    _scatter_panel(ax_det, "detection_roc_auc")
    _style_ax(ax_det)
    ax_det.set_ylim(0.70, 1.07)
    ax_det.set_ylabel("Detection (probe ROC-AUC)", fontsize=18, fontweight="bold", labelpad=10)
    ax_det.set_title("(a)  Detection efficacy", fontsize=22, fontweight="bold", pad=14)
    rho_s = f"{rho_layer:.2f}" if np.isfinite(rho_layer) else "—"
    p_s   = f"{p_layer:.3f}"   if np.isfinite(p_layer)   else "—"
    ax_det.text(0.97, 0.04, f"Spearman ρ = {rho_s},  p = {p_s}",
                transform=ax_det.transAxes, ha="right", va="bottom",
                fontsize=12, color="#111111")

    # right — transfer estimate
    _scatter_panel(ax_tr, "backdoor_transfer_estimate")
    _style_ax(ax_tr)
    y_pad = 0.012
    ax_tr.set_ylim(t_min - y_pad, t_max + y_pad)
    ax_tr.set_ylabel("Backdoor transfer estimate", fontsize=18, fontweight="bold", labelpad=10)
    ax_tr.set_title("(b)  Backdoor transfer", fontsize=22, fontweight="bold", pad=14)
    rho_te_s = f"{rho_te:.2f}" if np.isfinite(rho_te) else "—"
    p_te_s   = f"{p_te:.3f}"   if np.isfinite(p_te)   else "—"
    ax_tr.text(0.97, 0.04, f"Spearman ρ = {rho_te_s},  p = {p_te_s}",
               transform=ax_tr.transAxes, ha="right", va="bottom",
               fontsize=12, color="#111111")

    # shared legend below both panels
    ranks = sorted({r["rank"] for r in rows})
    leg_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_rank_color(rk),
               markeredgecolor="#111111", markeredgewidth=2.0,
               markersize=16, linestyle="None",
               label=f"LoRA rank  r = {rk}")
        for rk in ranks
    ]
    fig.legend(handles=leg_handles, loc="lower center", ncol=3,
               frameon=True, fontsize=16, borderpad=1.0,
               bbox_to_anchor=(0.5, -0.14))

    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {OUT_PNG}")
    print(f"Saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
