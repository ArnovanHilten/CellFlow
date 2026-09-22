"""Plot a CellFlow PCA×LR grid search from its per-run cell-eval outputs.

Reads every ``<results_dir>/*/final_test/agg_results.csv`` (written by
run_state_toml.py), parses ``n_pca`` and ``lr`` from the run-folder name
(``CellFlow_embedding_pca{N}_lr{LR}_...``), takes each run's mean-aggregate
metrics, and produces:

  - ``gridsearch_summary.csv`` : one tidy row per run (n_pca, lr, all metrics)
  - ``gridsearch_lines.png``   : each metric vs n_pca, one line per LR
  - ``gridsearch_heatmaps.png``: n_pca × lr heatmap per metric

Usage
-----
python scripts/plot_gridsearch.py \
    --results-dir /dcai/.../results/gridsearch \
    --out-dir     /dcai/.../results/gridsearch
"""

import argparse
import glob
import math
import os
import re

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# metric-name substrings where LOWER is better (else higher assumed)
_LOWER_BETTER = ("mae", "mse", "rmse", "error", "distance", "energy", "l2", "l1", "wasserstein")


def _cfg_from_run(run_name):
    """Extract (n_pca:int, lr:str) from a run-folder name, or None."""
    m = re.search(r"pca(\d+)_lr([0-9.eE+-]+)", run_name)
    return (int(m.group(1)), m.group(2)) if m else None


def load_grid(results_dir, eval_subdir="final_val"):
    """Collect mean metrics from every run's <eval_subdir>/agg_results.csv into a frame."""
    rows = []
    aggs = sorted(glob.glob(os.path.join(results_dir, "*", eval_subdir, "agg_results.csv")))
    if not aggs:
        raise FileNotFoundError(
            f"No */{eval_subdir}/agg_results.csv under {results_dir}. "
            f"Check the path, that runs finished and cell-eval succeeded, and that "
            f"validation metrics exist (re-run with --eval_only if only final_test is present)."
        )
    for agg in aggs:
        run = os.path.basename(os.path.dirname(os.path.dirname(agg)))
        cfg = _cfg_from_run(run)
        if cfg is None:
            print(f"  skip (can't parse pca/lr): {run}")
            continue
        n_pca, lr = cfg
        df = pd.read_csv(agg)
        if "statistic" in df.columns and (df["statistic"] == "mean").any():
            srow = df[df["statistic"] == "mean"].iloc[0]
        else:
            srow = df.iloc[0]  # fall back to the first row
        rec = {"run": run, "n_pca": n_pca, "lr": lr}
        for c in df.columns:
            if c == "statistic":
                continue
            try:
                rec[c] = float(srow[c])
            except (ValueError, TypeError):
                pass
        rows.append(rec)
    grid = pd.DataFrame(rows).sort_values(["lr", "n_pca"]).reset_index(drop=True)
    return grid


def _metric_cols(grid):
    return [c for c in grid.columns if c not in ("run", "n_pca", "lr")]


def _best(grid, metric):
    lower = any(k in metric.lower() for k in _LOWER_BETTER)
    idx = grid[metric].idxmin() if lower else grid[metric].idxmax()
    r = grid.loc[idx]
    return r["n_pca"], r["lr"], r[metric], ("min" if lower else "max")


def plot_lines(grid, metrics, out_path):
    """Line plot of each metric vs n_pca, one line per LR."""
    ncol = min(3, len(metrics))
    nrow = math.ceil(len(metrics) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    lrs = sorted(grid["lr"].unique())
    for ax, metric in zip(axes.flat, metrics, strict=False):
        for lr in lrs:
            sub = grid[grid["lr"] == lr].sort_values("n_pca")
            ax.plot(sub["n_pca"], sub[metric], marker="o", label=f"lr={lr}")
        n_pca, lr_b, val, direction = _best(grid, metric)
        ax.set_title(f"{metric}\nbest ({direction}): pca={n_pca}, lr={lr_b} → {val:.4g}", fontsize=9)
        ax.set_xlabel("n_pca_components")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_path}")


def plot_heatmaps(grid, metrics, out_path):
    """n_pca × lr heatmap per metric."""
    ncol = min(3, len(metrics))
    nrow = math.ceil(len(metrics) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.6 * nrow), squeeze=False)
    for ax, metric in zip(axes.flat, metrics, strict=False):
        piv = grid.pivot_table(index="lr", columns="n_pca", values=metric)
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(piv.columns)), piv.columns, fontsize=7)
        ax.set_yticks(range(len(piv.index)), piv.index, fontsize=7)
        ax.set_xlabel("n_pca")
        ax.set_ylabel("lr")
        ax.set_title(metric, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_path}")


def main():
    """CLI entry: load the grid, print best configs, write summary + figures."""
    p = argparse.ArgumentParser(description="Plot a CellFlow PCA×LR grid search")
    p.add_argument("--results-dir", required=True, help="Dir containing the per-run folders")
    p.add_argument("--eval-subdir", default="final_val",
                   help="Which split to read: final_val (selection) or final_test (reporting)")
    p.add_argument("--out-dir", default=None, help="Where to write outputs (default: results-dir)")
    p.add_argument("--metrics", default="", help="Comma-separated metric columns to plot (default: all)")
    args = p.parse_args()
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    grid = load_grid(args.results_dir, args.eval_subdir)
    grid.to_csv(os.path.join(out_dir, "gridsearch_summary.csv"), index=False)
    print(f"Loaded {len(grid)} runs. Columns: {list(grid.columns)}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] or _metric_cols(grid)
    metrics = [m for m in metrics if m in grid.columns]
    if not metrics:
        print("No numeric metric columns found to plot.")
        return

    print("\nBest config per metric:")
    for m in metrics:
        n_pca, lr, val, direction = _best(grid, m)
        print(f"  {m:24s} {direction}: n_pca={n_pca:<4} lr={lr:<6} = {val:.4g}")

    plot_lines(grid, metrics, os.path.join(out_dir, "gridsearch_lines.png"))
    plot_heatmaps(grid, metrics, os.path.join(out_dir, "gridsearch_heatmaps.png"))


if __name__ == "__main__":
    main()
