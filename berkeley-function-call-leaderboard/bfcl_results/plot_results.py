"""Plot BFCL leaderboard CSVs for a subset of 1B-class models.

Reads each data_*.csv in the default score directory (and optionally a second
one passed on the CLI), merges their rows, keeps only the rows whose Model is
in MODELS, and writes one PNG per CSV (named <csv_stem>.png).

Usage: python plot_results.py [score_dir]
  score_dir is a second score directory to merge with the default one.
  Absolute paths are used as-is; relative paths resolve against bfcl_results/.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CATS_PER_ROW = 4  # max categories (subplot x-axis groups) per subplot row.

DEFAULT_DIR = Path(__file__).parent / "2025-12-16" / "score"

SKIP_CSVS = {"data_format_sensitivity.csv"}

# 1B-class models to plot. Names must match the "Model" column verbatim.
MODELS = [
    "Arch-Agent-1.5B",
    "Falcon3-1B-Instruct (FC)",
    "Gemma-3-1b-it (Prompt)",
    "Hammer2.1-1.5b (FC)",
    "Llama-3.2-1B-Instruct (FC)",
    "Qwen3-1.7B (FC)",
    "xLAM-2-1b-fc-r (FC)",
]

# Columns to ignore when picking which columns to plot — non-score metadata
# plus "AST Summary", which duplicates the Live/Non-Live Overall Acc column in
# data_live.csv and data_non_live.csv.
META_COLS = {
    "Rank",
    "Model",
    "Model Link",
    "Organization",
    "License",
    "Total Cost ($)",
    "Latency Mean (s)",
    "Latency Standard Deviation (s)",
    "Latency 95th Percentile (s)",
    "AST Summary",
}


def to_float(x):
    if isinstance(x, str):
        x = x.strip().rstrip("%")
        if x in {"", "N/A"}:
            return float("nan")
        try:
            return float(x)
        except ValueError:
            return float("nan")
    return x


def load_merged(csv_name: str, dirs: list[tuple[Path, str]]) -> pd.DataFrame:
    """Read csv_name from each dir; baseline is filtered to MODELS, secondary
    dirs (any with a non-empty suffix) keep every row and get their Model
    tagged with ":<suffix>"."""
    frames = []
    for d, suffix in dirs:
        path = d / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if suffix:
            df["Model"] = df["Model"].astype(str) + f":{suffix}"
        else:
            df = df[df["Model"].isin(MODELS)].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def order_models(df: pd.DataFrame) -> list[str]:
    """Group each baseline model with its suffixed variants, then append any
    extra secondary-dir models not derived from MODELS."""
    present = list(df["Model"].astype(str))
    seen: set[str] = set()
    ordered: list[str] = []
    for m in MODELS:
        for x in present:
            if (x == m or x.startswith(m + ":")) and x not in seen:
                ordered.append(x)
                seen.add(x)
    for x in present:
        if x not in seen:
            ordered.append(x)
            seen.add(x)
    return ordered


def plot_csv(df: pd.DataFrame, csv_name: str, output: Path) -> None:
    if df.empty:
        print(f"skip {csv_name}: no rows")
        return
    model_order = order_models(df)

    value_cols = [c for c in df.columns if c not in META_COLS]
    for c in value_cols:
        df[c] = df[c].map(to_float)
    # Drop columns where every selected model is N/A so we don't waste x-axis space.
    value_cols = [c for c in value_cols if df[c].notna().any()]

    matrix = df.set_index("Model")[value_cols]
    matrix = matrix.reindex([m for m in model_order if m in matrix.index])

    n_models = max(1, len(matrix))
    n_rows = max(1, math.ceil(len(value_cols) / CATS_PER_ROW))
    fig_width = max(10, 0.55 * CATS_PER_ROW * n_models)
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(fig_width, 4.5 * n_rows + 1.5),
                             squeeze=False)
    fig.suptitle(f"{Path(csv_name).stem} — 1B-class models")

    legend_handles = None
    legend_labels = None
    for i in range(n_rows):
        cats = value_cols[i * CATS_PER_ROW:(i + 1) * CATS_PER_ROW]
        sub = matrix[cats].T  # rows=categories, columns=models (same order in every subplot → consistent colors)
        ax = axes[i, 0]
        sub.plot.bar(ax=ax, width=0.85, legend=False)
        ax.set_ylabel("score (%)")
        ax.set_xlabel("")
        ax.set_ylim(0, 110)
        # Label each bar with its score; NaN bars get an "N/A" label at the baseline
        # (bar_label drops labels whose y-coordinate is NaN, so we place them manually).
        for col_idx, container in enumerate(ax.containers):
            values = sub.iloc[:, col_idx].tolist()
            for rect, val in zip(container.patches, values):
                x = rect.get_x() + rect.get_width() / 2
                if pd.isna(val):
                    ax.text(x, 1, "N/A", ha="center", va="bottom",
                            fontsize=7, color="grey")
                else:
                    ax.text(x, val + 1, f"{val:.0f}", ha="center", va="bottom",
                            fontsize=7)
        # Pad short last rows so bar widths stay consistent with full rows.
        if n_rows > 1:
            ax.set_xlim(-0.5, CATS_PER_ROW - 0.5)
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center",
                   ncol=2, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output}")


def resolve_score_dir(arg: str) -> Path:
    p = Path(arg)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "score_dir",
        nargs="?",
        type=resolve_score_dir,
        help="extra score dir to merge with the default; absolute or relative to bfcl_results/",
    )
    args = parser.parse_args()

    dirs: list[tuple[Path, str]] = [(DEFAULT_DIR, "")]
    if args.score_dir is not None:
        if not args.score_dir.is_dir():
            parser.error(f"score_dir does not exist: {args.score_dir}")
        dirs.append((args.score_dir, args.score_dir.name))

    output_dir = args.score_dir if args.score_dir is not None else DEFAULT_DIR

    csv_names = sorted({p.name for d, _ in dirs for p in d.glob("*.csv")} - SKIP_CSVS)
    for name in csv_names:
        df = load_merged(name, dirs)
        if df.empty:
            continue
        plot_csv(df, name, output_dir / (Path(name).stem + ".png"))


if __name__ == "__main__":
    main()
