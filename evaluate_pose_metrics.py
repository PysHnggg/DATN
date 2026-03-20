#!/usr/bin/env python3
"""
Evaluate pose-estimation metrics from pose_log.csv.

Updated for the newer run.py export format, including:
- marker measurement columns from plane-fit + ray/plane intersection
- optional marker GT-based error metrics
- robust summaries for plotting / paper tables

Outputs:
- clean_pose_metrics.csv
- summary_overall.csv
- summary_by_class.csv
- summary_by_distance_bin.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


NUMERIC_CANDIDATES = [
    "frame", "object_id",
    "cx", "cy", "cz", "gt_cx", "gt_cy", "gt_cz",
    "trans_err_m", "trans_err_cm", "dx", "dy", "dz",
    "trans_err_normal_cm", "trans_err_tangent_cm",
    "rot_err_deg", "normal_err_deg", "algorithm_rot_err_deg",
    "num_points", "valid_depth_ratio", "z_median_m", "z_std_m",
    "eigval1", "eigval2", "eigval3", "linearity", "planarity", "scattering",
    "length_m", "width_m", "height_m", "dim_max_m", "dim_mid_m", "dim_min_m",
    "marker_measured_width_cm", "marker_measured_height_cm",
    "marker_measured_mean_edge_cm", "marker_measured_std_edge_cm",
    "marker_size_gt_cm", "marker_size_err_cm",
]
NUMERIC_CANDIDATES += [f"r{i}{j}" for i in range(3) for j in range(3)]
NUMERIC_CANDIDATES += [f"gt_r{i}{j}" for i in range(3) for j in range(3)]

REQUIRED_ROT_EST = [f"r{i}{j}" for i in range(3) for j in range(3)]
REQUIRED_ROT_GT = [f"gt_r{i}{j}" for i in range(3) for j in range(3)]
DISTANCE_BINS = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.5, np.inf]
DISTANCE_LABELS = ["0-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0", "1.0-1.5", "1.5-2.5", ">2.5"]


def rotation_error_deg_from_row(row: pd.Series) -> float:
    """Geodesic distance on SO(3). Only valid when marker_on_object."""
    if "marker_on_object" in row.index:
        mo = row["marker_on_object"]
        if mo is False or str(mo).lower() in ("false", "0", ""):
            return np.nan
    Re = np.array([[row[f"r{i}{j}"] for j in range(3)] for i in range(3)], dtype=float)
    Rg = np.array([[row[f"gt_r{i}{j}"] for j in range(3)] for i in range(3)], dtype=float)
    if not np.isfinite(Re).all() or not np.isfinite(Rg).all():
        return np.nan
    R_delta = Rg @ Re.T
    val = (np.trace(R_delta) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return math.degrees(math.acos(val))


def ensure_metrics(df: pd.DataFrame, marker_size_cm: Optional[float] = None) -> pd.DataFrame:
    df = df.copy()

    for c in NUMERIC_CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def _col_empty(col: str) -> bool:
        """True if column is absent or all-NaN (i.e. needs to be computed)."""
        return col not in df.columns or df[col].isna().all()

    if {"cx", "cy", "cz", "gt_cx", "gt_cy", "gt_cz"}.issubset(df.columns):
        if _col_empty("dx"):
            df["dx"] = df["cx"] - df["gt_cx"]
        if _col_empty("dy"):
            df["dy"] = df["cy"] - df["gt_cy"]
        if _col_empty("dz"):
            df["dz"] = df["cz"] - df["gt_cz"]
        if _col_empty("trans_err_m"):
            df["trans_err_m"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2 + df["dz"] ** 2)
        if _col_empty("trans_err_cm"):
            df["trans_err_cm"] = df["trans_err_m"] * 100.0

    if all(c in df.columns for c in REQUIRED_ROT_EST + REQUIRED_ROT_GT) and _col_empty("rot_err_deg"):
        df["rot_err_deg"] = df.apply(rotation_error_deg_from_row, axis=1)

    if {"eigval1", "eigval2", "eigval3"}.issubset(df.columns):
        eps = 1e-12
        if _col_empty("linearity"):
            df["linearity"] = (df["eigval1"] - df["eigval2"]) / (df["eigval1"] + eps)
        if _col_empty("planarity"):
            df["planarity"] = (df["eigval2"] - df["eigval3"]) / (df["eigval1"] + eps)
        if _col_empty("scattering"):
            df["scattering"] = df["eigval3"] / (df["eigval1"] + eps)

    if {"length_m", "width_m", "height_m"}.issubset(df.columns):
        dims = np.sort(df[["length_m", "width_m", "height_m"]].to_numpy(dtype=float), axis=1)[:, ::-1]
        df["dim_max_m"] = dims[:, 0]
        df["dim_mid_m"] = dims[:, 1]
        df["dim_min_m"] = dims[:, 2]

    # Optional object-vs-marker debug columns (kept only if user asks for them)
    if marker_size_cm is not None:
        marker_m = marker_size_cm / 100.0
        for col in ["length_m", "width_m"]:
            if col in df.columns:
                df[f"{col}_err_cm_vs_marker"] = (df[col] - marker_m).abs() * 100.0
        if "marker_size_gt_cm" not in df.columns:
            df["marker_size_gt_cm"] = marker_size_cm

    # Marker measurement error from actual measured marker size
    if "marker_size_gt_cm" in df.columns and "marker_measured_mean_edge_cm" in df.columns:
        if "marker_size_err_cm" not in df.columns:
            df["marker_size_err_cm"] = (df["marker_measured_mean_edge_cm"] - df["marker_size_gt_cm"]).abs()

    # Prefer GT distance for binning if available
    if "distance_bin_m" not in df.columns:
        if "gt_cz" in df.columns and df["gt_cz"].notna().any():
            df["distance_bin_m"] = pd.cut(df["gt_cz"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS, include_lowest=True)
        elif "cz" in df.columns:
            df["distance_bin_m"] = pd.cut(df["cz"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS, include_lowest=True)

    return df


def summarize_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append({
            "metric": c,
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
            "q05": float(s.quantile(0.05)),
            "q95": float(s.quantile(0.95)),
        })
    return pd.DataFrame(rows)


def grouped_summary(df: pd.DataFrame, group_col: str, metrics: List[str]) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(group_col, dropna=False):
        entry = {group_col: key, "n": int(len(g))}
        for m in metrics:
            if m not in g.columns:
                continue
            s = pd.to_numeric(g[m], errors="coerce").dropna()
            if len(s) == 0:
                entry[f"{m}_mean"] = np.nan
                entry[f"{m}_std"] = np.nan
                entry[f"{m}_median"] = np.nan
            else:
                entry[f"{m}_mean"] = float(s.mean())
                entry[f"{m}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
                entry[f"{m}_median"] = float(s.median())
        out.append(entry)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to pose_log.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--marker-size-cm", type=float, default=None, help="Optional GT marker side length in cm")
    ap.add_argument("--class-col", default="class_name", help="Class column name")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df = ensure_metrics(df, marker_size_cm=args.marker_size_cm)

    metric_cols = [
        "trans_err_cm", "trans_err_normal_cm", "trans_err_tangent_cm",
        "rot_err_deg", "normal_err_deg", "algorithm_rot_err_deg",
        "dx", "dy", "dz",
        "num_points", "valid_depth_ratio", "z_median_m", "z_std_m",
        "eigval1", "eigval2", "eigval3", "linearity", "planarity", "scattering",
        "length_m", "width_m", "height_m", "dim_max_m", "dim_mid_m", "dim_min_m",
        "marker_measured_width_cm", "marker_measured_height_cm",
        "marker_measured_mean_edge_cm", "marker_measured_std_edge_cm", "marker_size_gt_cm", "marker_size_err_cm",
        "length_m_err_cm_vs_marker", "width_m_err_cm_vs_marker",
    ]

    clean_path = outdir / "clean_pose_metrics.csv"
    df.to_csv(clean_path, index=False)

    overall = summarize_numeric(df, metric_cols)
    overall.to_csv(outdir / "summary_overall.csv", index=False)

    if args.class_col in df.columns:
        by_class = grouped_summary(df, args.class_col, [m for m in metric_cols if m in df.columns])
        by_class.to_csv(outdir / "summary_by_class.csv", index=False)
    else:
        by_class = None

    if "distance_bin_m" in df.columns:
        by_dist = grouped_summary(
            df,
            "distance_bin_m",
            [m for m in [
                "trans_err_cm", "trans_err_tangent_cm", "trans_err_normal_cm",
                "rot_err_deg", "normal_err_deg", "algorithm_rot_err_deg",
                "num_points", "valid_depth_ratio",
                "marker_measured_mean_edge_cm", "marker_size_err_cm"
            ] if m in df.columns],
        )
        by_dist.to_csv(outdir / "summary_by_distance_bin.csv", index=False)
    else:
        by_dist = None

    print("=" * 72)
    print("POSE EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Input rows: {len(df)}")
    print(f"Saved cleaned metrics: {clean_path}")
    print()

    show_metrics = [
        m for m in [
            "trans_err_cm", 
            "trans_err_normal_cm", 
            "trans_err_tangent_cm",
            "rot_err_deg", "normal_err_deg", "algorithm_rot_err_deg", "algorithm_rot_err_deg",
            "marker_measured_mean_edge_cm", "marker_size_err_cm",
            "num_points", "valid_depth_ratio",
            "linearity", "planarity", "scattering",
            "length_m", "width_m", "height_m",
        ] if m in df.columns
    ]
    for m in show_metrics:
        s = pd.to_numeric(df[m], errors="coerce").dropna()
        if len(s) == 0:
            continue
        print(
            f"{m:>26s}: mean={s.mean():.4f}, std={s.std(ddof=1) if len(s) > 1 else 0.0:.4f}, "
            f"median={s.median():.4f}, min={s.min():.4f}, max={s.max():.4f}"
        )

    if by_class is not None:
        print("\nClasses found:", ", ".join(map(str, by_class[args.class_col].tolist())))
    if by_dist is not None:
        print("Distance-bin summary written.")
    print("Summary CSV files written to:", outdir)


if __name__ == "__main__":
    main()
