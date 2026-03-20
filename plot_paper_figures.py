#!/usr/bin/env python3
"""
Generate paper-ready matplotlib figures from clean_pose_metrics.csv.

Updated to support marker measurement figures from the newer run.py export.
Outputs PNG + PDF versions when possible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


def save_both(fig, outbase: Path) -> None:
    fig.tight_layout()
    fig.savefig(str(outbase.with_suffix(".png")), bbox_inches="tight")
    fig.savefig(str(outbase.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def hist_plot(df: pd.DataFrame, col: str, outbase: Path, title: str, xlabel: str, bins: int = 30) -> None:
    if col not in df.columns:
        return
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(s, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    save_both(fig, outbase)


def scatter_plot(df: pd.DataFrame, x: str, y: str, outbase: Path, title: str, xlabel: str, ylabel: str) -> None:
    if x not in df.columns or y not in df.columns:
        return
    d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.scatter(d[x], d[y], s=14)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_both(fig, outbase)


def box_plot_by_group(df: pd.DataFrame, group_col: str, value_col: str, outbase: Path, title: str, ylabel: str) -> None:
    if group_col not in df.columns or value_col not in df.columns:
        return
    data = []
    labels = []
    for key, g in df.groupby(group_col, dropna=False):
        s = pd.to_numeric(g[value_col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        labels.append(str(key))
        data.append(s.to_numpy())
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(group_col)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    save_both(fig, outbase)


def line_plot_distance_summary(df: pd.DataFrame, outbase: Path) -> None:
    if "distance_bin_m" not in df.columns:
        return
    cols = [c for c in ["trans_err_cm", "rot_err_deg", "marker_size_err_cm"] if c in df.columns]
    if not cols:
        return
    grouped = df.groupby("distance_bin_m", observed=False)[cols].mean(numeric_only=True)
    if grouped.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(grouped.index))
    for col in grouped.columns:
        ax.plot(x, grouped[col].to_numpy(), marker="o", label=col)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in grouped.index], rotation=20, ha="right")
    ax.set_title("Mean error by distance bin")
    ax.set_xlabel("Distance bin (m)")
    ax.set_ylabel("Error")
    ax.legend()
    save_both(fig, outbase)


def bar_pca_metrics(df: pd.DataFrame, outbase: Path) -> None:
    cols = [c for c in ["linearity", "planarity", "scattering"] if c in df.columns]
    if not cols:
        return
    vals = [pd.to_numeric(df[c], errors="coerce").dropna().mean() for c in cols]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.bar(cols, vals)
    ax.set_title("Average PCA shape metrics")
    ax.set_ylabel("Mean value")
    save_both(fig, outbase)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to clean_pose_metrics.csv")
    ap.add_argument("--outdir", required=True, help="Output figure directory")
    ap.add_argument("--class-col", default="class_name")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    hist_plot(df, "trans_err_cm", outdir / "fig_translation_error_hist", "Translation error distribution", "Translation error (cm)")
    hist_plot(df, "rot_err_deg", outdir / "fig_rotation_error_hist", "Rotation error distribution", "Rotation error (deg)")
    hist_plot(df, "num_points", outdir / "fig_num_points_hist", "Point-cloud size distribution", "Number of points")
    hist_plot(df, "valid_depth_ratio", outdir / "fig_valid_depth_ratio_hist", "Valid depth ratio distribution", "Valid depth ratio")
    hist_plot(df, "marker_measured_mean_edge_cm", outdir / "fig_marker_measurement_hist", "Measured marker size distribution", "Measured marker size (cm)")
    hist_plot(df, "marker_size_err_cm", outdir / "fig_marker_size_error_hist", "Marker size error distribution", "Marker size error (cm)")

    x_depth = None
    if "gt_cz" in df.columns and pd.to_numeric(df["gt_cz"], errors="coerce").notna().any():
        x_depth = "gt_cz"
    elif "cz" in df.columns:
        x_depth = "cz"

    if x_depth is not None:
        scatter_plot(df, x_depth, "trans_err_cm", outdir / "fig_translation_error_vs_depth", "Translation error vs distance", "Distance Z (m)", "Translation error (cm)")
        scatter_plot(df, x_depth, "rot_err_deg", outdir / "fig_rotation_error_vs_depth", "Rotation error vs distance", "Distance Z (m)", "Rotation error (deg)")
        scatter_plot(df, x_depth, "marker_size_err_cm", outdir / "fig_marker_error_vs_distance", "Marker size error vs distance", "Distance Z (m)", "Marker size error (cm)")
        scatter_plot(df, x_depth, "marker_measured_mean_edge_cm", outdir / "fig_marker_measurement_vs_distance", "Measured marker size vs distance", "Distance Z (m)", "Measured marker size (cm)")

    if args.class_col in df.columns:
        box_plot_by_group(df, args.class_col, "trans_err_cm", outdir / "fig_translation_error_by_class", "Translation error by class", "Translation error (cm)")
        box_plot_by_group(df, args.class_col, "rot_err_deg", outdir / "fig_rotation_error_by_class", "Rotation error by class", "Rotation error (deg)")
        box_plot_by_group(df, args.class_col, "marker_size_err_cm", outdir / "fig_marker_size_error_by_class", "Marker size error by class", "Marker size error (cm)")

    line_plot_distance_summary(df, outdir / "fig_error_by_distance_bin")
    bar_pca_metrics(df, outdir / "fig_pca_shape_metrics")

    print(f"Figures saved to: {outdir}")


if __name__ == "__main__":
    main()
