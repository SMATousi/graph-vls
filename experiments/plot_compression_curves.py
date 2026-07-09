"""Plot reconstruction F1 vs. k, d, and compression ratio for the Phase 3
compression sweep.

Reads results/compression/{cora,citeseer,pubmed}.csv (written by
compression_sweep.py) and produces three figures:
  - f1_vs_k.png, f1_vs_d.png: small multiples (one panel per dataset), F1
    against the swept parameter, colored by the other parameter.
  - f1_vs_ratio.png: the actual rate-distortion curve -- F1 against
    dim_compression_ratio and edge_compression_ratio, all three datasets
    overlaid (colored by dataset, since here the datasets are directly
    comparable on a common, dimensionless x-axis).
All three draw the 0.90 fidelity floor as a reference line so the "none of
the three datasets reach it" finding in README.md is visible directly, not
just asserted in prose.

Usage:
    python experiments/plot_compression_curves.py
"""

import csv
import os

import matplotlib.pyplot as plt

DATASETS = ["cora", "citeseer", "pubmed"]
TITLES = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}

# Sequential blue ramp (specs/phase3 dataviz convention), light->dark,
# assigned in fixed magnitude order -- see references/palette.md.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

# Categorical palette, first 3 slots in fixed order -- dataset identity, not
# magnitude, so this uses the categorical rule instead of the sequential ramp.
DATASET_COLOR = {"cora": "#2a78d6", "citeseer": "#1baf7a", "pubmed": "#eda100"}

SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"

FIDELITY_FLOOR = 0.90
RESULTS_DIR = "results/compression"


def _load(dataset: str) -> list[dict]:
    path = os.path.join(RESULTS_DIR, f"{dataset}.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["latent_dim"] = int(r["latent_dim"])
        r["k"] = int(r["k"])
        r["reconstruction_f1"] = float(r["reconstruction_f1"])
        r["dim_compression_ratio"] = float(r["dim_compression_ratio"])
        r["edge_compression_ratio"] = float(r["edge_compression_ratio"])
    return rows


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED_INK, labelsize=9)


LEGEND_TITLES = {"latent_dim": "d", "k": "k"}


def _plot_grid(
    x_key: str,
    series_key: str,
    series_values: list[int],
    x_values: list[int],
    out_path: str,
    x_label: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dataset in zip(axes, DATASETS):
        rows = _load(dataset)
        _style_axis(ax)

        for color, series_val in zip(RAMP, series_values):
            xs, ys = [], []
            for xv in x_values:
                match = [
                    r for r in rows if r[series_key] == series_val and r[x_key] == xv
                ]
                if match:
                    xs.append(xv)
                    ys.append(match[0]["reconstruction_f1"])
            ax.plot(
                range(len(xs)),
                ys,
                color=color,
                linewidth=2,
                marker="o",
                markersize=6,
                label=str(series_val),
                zorder=3,
            )

        ax.axhline(
            FIDELITY_FLOOR, color=MUTED_INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2
        )
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([str(v) for v in x_values])
        ax.set_title(TITLES[dataset], color=PRIMARY_INK, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(x_label, color=SECONDARY_INK, fontsize=10)

    axes[0].set_ylabel("Reconstruction F1", color=SECONDARY_INK, fontsize=10)
    axes[0].set_ylim(0.60, 0.95)
    axes[-1].text(
        len(x_values) - 1,
        FIDELITY_FLOOR + 0.006,
        "0.90 fidelity floor",
        color=MUTED_INK,
        fontsize=9,
        ha="right",
        va="bottom",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title=LEGEND_TITLES[series_key],
        loc="lower center",
        ncol=len(series_values),
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        labelcolor=SECONDARY_INK,
        title_fontproperties={"weight": "bold"},
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _plot_ratio_scatter(out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.patch.set_facecolor(SURFACE)

    panels = [
        ("dim_compression_ratio", "d / F  (dimension ratio, log scale)", axes[0], False),
        ("edge_compression_ratio", "|A_z| / |E|  (edge ratio, log scale)", axes[1], True),
    ]

    for ratio_key, x_label, ax, show_unity_line in panels:
        _style_axis(ax)
        ax.set_xscale("log")

        for dataset in DATASETS:
            rows = _load(dataset)
            xs = [r[ratio_key] for r in rows]
            ys = [r["reconstruction_f1"] for r in rows]
            ax.scatter(
                xs,
                ys,
                s=42,
                color=DATASET_COLOR[dataset],
                alpha=0.75,
                edgecolors=SURFACE,
                linewidths=0.5,
                label=TITLES[dataset],
                zorder=3,
            )

        if show_unity_line:
            ax.axvline(1.0, color=MUTED_INK, linewidth=1.0, linestyle=(0, (1, 2)), zorder=2)
            ax.text(
                1.0,
                0.605,
                "  same density as input",
                color=MUTED_INK,
                fontsize=8,
                rotation=90,
                va="bottom",
                ha="left",
            )

        ax.axhline(
            FIDELITY_FLOOR, color=MUTED_INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2
        )
        ax.set_xlabel(x_label, color=SECONDARY_INK, fontsize=10)
        ax.set_ylim(0.60, 0.95)

    axes[0].set_ylabel("Reconstruction F1", color=SECONDARY_INK, fontsize=10)
    axes[1].text(
        axes[1].get_xlim()[1],
        FIDELITY_FLOOR + 0.006,
        "0.90 fidelity floor  ",
        color=MUTED_INK,
        fontsize=9,
        ha="right",
        va="bottom",
    )
    fig.suptitle(
        "Reconstruction fidelity vs. compression ratio (all 36 (d,k) points per dataset)",
        color=PRIMARY_INK,
        fontsize=12,
        fontweight="bold",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        labelcolor=SECONDARY_INK,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    _plot_grid(
        x_key="k",
        series_key="latent_dim",
        series_values=[4, 8, 16, 32, 64, 128],
        x_values=[1, 2, 3, 5, 10, 20],
        out_path=os.path.join(RESULTS_DIR, "f1_vs_k.png"),
        x_label="k (latent graph top-k)",
    )
    _plot_grid(
        x_key="latent_dim",
        series_key="k",
        series_values=[1, 2, 3, 5, 10, 20],
        x_values=[4, 8, 16, 32, 64, 128],
        out_path=os.path.join(RESULTS_DIR, "f1_vs_d.png"),
        x_label="d (latent dimension)",
    )
    _plot_ratio_scatter(os.path.join(RESULTS_DIR, "f1_vs_ratio.png"))


if __name__ == "__main__":
    main()
