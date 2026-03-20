Evaluation toolkit for pose_log.csv
===================================

Files
-----
1. evaluate_pose_metrics.py
   - Reads pose_log.csv
   - Computes / verifies metrics
   - Writes clean_pose_metrics.csv, summary_overall.csv, summary_by_class.csv, summary_by_distance_bin.csv

2. plot_paper_figures.py
   - Reads clean_pose_metrics.csv
   - Generates paper-ready figures using matplotlib

3. ieee_icra_iros_results_template.tex
   - LaTeX table templates for IEEE / ICRA / IROS style papers

Typical workflow
----------------
1) Run your acquisition / experiment script to create:
   outputs/pose/pose_log.csv

2) Evaluate metrics:
   python evaluate_pose_metrics.py --input outputs/pose/pose_log.csv --outdir outputs/eval --marker-size-cm 4.0

3) Generate figures:
   python plot_paper_figures.py --input outputs/eval/clean_pose_metrics.csv --outdir outputs/eval/figures

Recommended columns in pose_log.csv
-----------------------------------
Estimate pose:
- cx, cy, cz
- r00..r22
Ground truth pose:
- gt_cx, gt_cy, gt_cz
- gt_r00..gt_r22
Quality metrics:
- num_points, valid_depth_ratio, z_median_m, z_std_m
PCA metrics:
- eigval1, eigval2, eigval3, linearity, planarity, scattering
Dimension metrics:
- length_m, width_m, height_m
Meta:
- class_name, object_id, frame

Main outputs
------------
- clean_pose_metrics.csv: cleaned row-level data for plotting
- summary_overall.csv: mean / std / min / max / quantiles
- summary_by_class.csv: grouped stats by class
- summary_by_distance_bin.csv: grouped stats by distance bin

Notes
-----
- If trans_err_m / trans_err_cm are missing but both estimate and GT positions exist,
  the script computes them automatically.
- If rot_err_deg is missing but both estimate and GT rotation matrices exist,
  the script computes it automatically.
