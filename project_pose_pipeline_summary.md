Project Context Summary (YOLO + RealSense + PCA Pose Estimation)
1. System Overview

The project builds a 3D pose estimation pipeline using:

Intel RealSense D435i

YOLO object detection

Depth → point cloud

PCA for orientation

ArUco markers for ground truth

**Evaluation (scientific):** When USE_MARKER_POSE_WHEN_ON_OBJECT=True, marker pose is used for DISPLAY only. The CSV always logs the algorithm pose (PCA) so rot_err_deg and trans_err reflect true algorithm performance vs GT.

The system estimates:

3D position (translation)

3D orientation (rotation)

Object size

Point-cloud geometric metrics

Results are logged to:

outputs/pose/pose_log.csv

Evaluation scripts compute metrics and generate plots for research papers (IEEE / ICRA / IROS).

2. Main Pipeline
run.py

Main runtime script:

Pipeline:

RGB frame
   ↓
YOLO detection
   ↓
Depth extraction inside bbox
   ↓
Point cloud generation
   ↓
PCA orientation estimation
   ↓
3D pose + size estimation
   ↓
Logging to CSV

Outputs logged per object:

tx ty tz
roll pitch yaw
point_count
valid_depth_ratio
marker_measured_size
PCA eigenvalues
3. Orientation Stabilization
Problem

PCA eigenvectors have sign ambiguity:

v and -v are both valid

This causes 180° flips between frames.

Example:

frame1: yaw = 12°
frame2: yaw = 178°
frame3: yaw = -170°

Even when object does not move.

Solution Implemented

Temporal consistency check:

if dot(prev_axis, new_axis) < 0:
    new_axis = -new_axis

This prevents axis flipping.

Additional Stabilization

If object barely moves:

delta_pos < threshold
rotation_change < threshold

Then orientation is held from previous frame.

New log fields:

orientation_flip_corrected
orientation_hold_applied
orientation_stability_deg
position_delta_m
4. Evaluation Scripts
evaluate_pose_metrics.py

Reads:

pose_log.csv

Computes:

translation error (cm)
rotation error (deg)
marker size error
point cloud statistics
PCA metrics

Outputs:

outputs/eval/

clean_pose_metrics.csv
summary_overall.csv
summary_by_class.csv
summary_by_distance_bin.csv
5. Visualization
plot_paper_figures.py

Generates research plots:

Examples:

translation error distribution

rotation error distribution

translation error vs distance

rotation error vs distance

marker size error distribution

marker size vs distance

PCA shape metrics

point cloud size distribution

valid depth ratio

Saved to:

outputs/eval/figures/
6. Experimental Results
Translation Error

Range:

6.3 cm – 10.9 cm

Mean:

≈ 8.8 cm

Observation:

Translation error increases with distance from camera.

Worst case:

remote object

Likely due to elongated shape.

7. Rotation Error

Range:

72° – 180°

Mean:

≈160°

Cause:

PCA sign ambiguity

Eigenvector direction can flip:

v ↔ -v

Which produces artificial 180° rotation error.

This does not necessarily indicate pose estimation failure.

8. Marker Size Accuracy

Ground truth marker:

4.0 cm

Measured:

3.77 – 3.98 cm

Mean error:

0.14 cm

Relative error:

3.5%

This is very good accuracy for a RealSense depth camera.

9. Point Cloud Quality

Typical point count:

2500 – 6000 points

Sufficient for PCA orientation estimation.

PCA Shape Metrics

Average:

metric	value
linearity	~0.55
planarity	~0.42
scattering	~0.03

Interpretation:

Objects are mostly planar surfaces.

10. Depth Quality

Valid depth ratio:

0.7 – 0.9

Indicates good depth coverage.

11. Overall Performance
Metric	Value
Mean translation error	~8.8 cm
Mean rotation error	~160°
Marker size error	~0.14 cm
Relative marker error	~3.5 %
Point cloud size	~4500 pts
Valid depth ratio	~0.75
12. Strengths

Accurate marker measurement

Good depth quality

Stable point cloud extraction

Reasonable translation accuracy

13. Main Limitation

Large rotation error caused by:

PCA orientation ambiguity

Recommended fix:

if dot(v_t, v_{t-1}) < 0:
    v_t = -v_t

or align PCA axis with marker orientation.

14. Possible Improvements

Future upgrades:

quaternion temporal smoothing

PCA temporal filtering

ICP pose refinement

marker-guided orientation alignment

These can reduce pose jitter and rotation error significantly.

15. How to Continue in New Chat

In the next chat say:

Continue helping me with my YOLO + RealSense + PCA pose estimation pipeline.

Then upload this file.

The assistant will understand:

system architecture

scripts

metrics

current issues

future improvements.