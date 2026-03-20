#!/usr/bin/env python3
"""
YOLO-3D Pose Estimation: RealSense depth + YOLO detection + PCA orientation.

Frame conventions (OpenCV/RealSense camera):
  - Camera: X right, Y down, Z into scene
  - Object (ArUco & PCA): X right, Y down, Z toward camera
  - R: object-to-camera, p_cam = R @ p_obj

Rotation error: geodesic distance on SO(3), angle = arccos((trace(R_gt @ R_est.T) - 1) / 2)
"""
import os
import csv
import time
import math
import itertools
import cv2
import json
import socket
import numpy as np
import pyrealsense2 as rs
from detection_model import ObjectDetector
from realsense_depth import RealSenseDepth
import dashboard

OUTDIR = "outputs"
os.makedirs(f"{OUTDIR}/frames", exist_ok=True)
os.makedirs(f"{OUTDIR}/depth", exist_ok=True)
os.makedirs(f"{OUTDIR}/clouds", exist_ok=True)
os.makedirs(f"{OUTDIR}/pose", exist_ok=True)
POSE_CSV = f"{OUTDIR}/pose/pose_log.csv"

# Set True if roll displays with wrong sign (e.g. tilt right shows negative roll)
RPY_ROLL_NEGATE = False

# Use RANSAC plane fitting for PCA (target: rot_err < 0.035 rad)
PCA_USE_RANSAC = True

# When marker is on object: use marker pose for DISPLAY only. For evaluation we always log
# algorithm pose (PCA) to get scientifically valid rot_err_deg and trans_err.
USE_MARKER_POSE_WHEN_ON_OBJECT = True

POSE_CSV_HEADER = [
    "frame", "timestamp", "object_index", "object_id", "class_id", "class_name", "confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "num_points", "valid_depth_ratio", "z_median_m", "z_std_m",
    "cx", "cy", "cz",
    "roll_deg", "pitch_deg", "yaw_deg",
    "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22",
    "length_m", "width_m", "height_m",
    "extent_axis0_m", "extent_axis1_m", "extent_axis2_m",
    "eigval1", "eigval2", "eigval3",
    "linearity", "planarity", "scattering",
    "gt_marker_id", "gt_distance_m",
    "gt_cx", "gt_cy", "gt_cz",
    "gt_roll_deg", "gt_pitch_deg", "gt_yaw_deg",
    "gt_r00", "gt_r01", "gt_r02", "gt_r10", "gt_r11", "gt_r12", "gt_r20", "gt_r21", "gt_r22",
    "marker_on_object",
    "trans_err_m", "trans_err_cm", "dx", "dy", "dz",
    "trans_err_normal_cm", "trans_err_tangent_cm",
    "rot_err_deg", "normal_err_deg", "algorithm_rot_err_deg",
    "marker_measured_width_cm", "marker_measured_height_cm", "marker_measured_mean_edge_cm",
    "marker_measured_std_edge_cm", "marker_size_gt_cm", "marker_size_err_cm",
    "cube_edge_est_cm", "cube_size_cm", "cube_size_label",
]


def init_pose_csv():
    if not os.path.exists(POSE_CSV):
        with open(POSE_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(POSE_CSV_HEADER)


def append_pose_row(row_dict):
    with open(POSE_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row_dict.get(col, "") for col in POSE_CSV_HEADER])


def create_isaac_udp_client(host="127.0.0.1", port=6000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((host, port))
    return sock


def send_cube_to_isaac(sock, frame_id, class_name, center_est, R_est, length_m, width_m, height_m,
                       cube_size_label, cube_size_cm, target_position=None):
    if sock is None:
        return
    if target_position is None:
        target_position = np.array([-0.3, -0.3, 0.12])
    msg = {
        "frame": int(frame_id),
        "name": str(class_name),
        "cube_size_label": str(cube_size_label) if cube_size_label else "",
        "cube_size_cm": float(cube_size_cm) if cube_size_cm != "" else 0.0,
        "position": {"x": float(center_est[0]), "y": float(center_est[1]), "z": float(center_est[2])},
        "rotation_matrix": [
            [float(R_est[0, 0]), float(R_est[0, 1]), float(R_est[0, 2])],
            [float(R_est[1, 0]), float(R_est[1, 1]), float(R_est[1, 2])],
            [float(R_est[2, 0]), float(R_est[2, 1]), float(R_est[2, 2])],
        ],
        "size_m": {"length": float(length_m), "width": float(width_m), "height": float(height_m)},
        "target_position": {"x": float(target_position[0]), "y": float(target_position[1]), "z": float(target_position[2])},
    }
    try:
        sock.send(json.dumps(msg).encode("utf-8"))
    except Exception:
        pass


def save_frame_bundle(frame_id, color_bgr, vis_bgr, depth_m, depth_color_bgr=None, pc_view_bgr=None):
    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{frame_id:06d}_{ts}"
    cv2.imwrite(f"{OUTDIR}/frames/{prefix}_rgb.png", color_bgr)
    cv2.imwrite(f"{OUTDIR}/frames/{prefix}_vis.png", vis_bgr)
    np.save(f"{OUTDIR}/depth/{prefix}_depth.npy", depth_m)
    if depth_color_bgr is not None:
        cv2.imwrite(f"{OUTDIR}/depth/{prefix}_depth_color.png", depth_color_bgr)
    else:
        d8 = np.clip(np.nan_to_num(depth_m, nan=0.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{OUTDIR}/depth/{prefix}_depth_color.png", cv2.applyColorMap(d8, cv2.COLORMAP_JET))
    if pc_view_bgr is not None:
        cv2.imwrite(f"{OUTDIR}/frames/{prefix}_pointcloud.png", pc_view_bgr)
    print(f"[Saved] {prefix}")


def save_ply_xyz(points_xyz, path):
    if points_xyz is None or len(points_xyz) == 0:
        return
    pts = points_xyz.astype(np.float32)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in pts:
            f.write(f"{x} {y} {z}\n")


def save_pca_axes_ply(points_xyz, center, R, axis_len=0.06, path="roi_pca.ply"):
    if points_xyz is None or len(points_xyz) == 0:
        return
    pts = points_xyz.astype(np.float32)
    c = center.astype(np.float32)
    total = pts.shape[0] + 4
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for x, y, z in pts:
            f.write(f"{x} {y} {z} 255 255 255\n")
        f.write(f"{c[0]} {c[1]} {c[2]} 255 255 0\n")
        for i, col in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            end = c + R[:, i].astype(np.float32) * axis_len
            f.write(f"{end[0]} {end[1]} {end[2]} {col[0]} {col[1]} {col[2]}\n")


def _overlay_rgba(dst_bgr, src_rgba, x, y):
    h, w = dst_bgr.shape[:2]
    sh, sw = src_rgba.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + sw), min(h, y + sh)
    if x1 >= x2 or y1 >= y2:
        return
    roi = dst_bgr[y1:y2, x1:x2]
    patch = src_rgba[(y1 - y):(y2 - y), (x1 - x):(x2 - x)]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    dst_bgr[y1:y2, x1:x2] = (roi.astype(np.float32) * (1.0 - alpha) + patch[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)


def rotate_rgba_expand(img_rgba, angle_deg):
    h, w = img_rgba.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(img_rgba, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def draw_text_along_edge(vis, p1, p2, text, offset_px=10, font_scale=0.45, thickness=1):
    x1, y1, x2, y2 = float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    angle = math.degrees(math.atan2(dy, dx))
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx, ny = -dy / L, dx / L
    tx, ty = mx + nx * offset_px, my + ny * offset_px
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    patch = np.zeros((th + base + pad * 2, tw + pad * 2, 4), dtype=np.uint8)
    cv2.putText(patch, text, (pad, pad + th), font, font_scale, (0, 0, 0, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(patch, text, (pad, pad + th), font, font_scale, (255, 255, 255, 255), thickness, cv2.LINE_AA)
    rot = rotate_rgba_expand(patch, angle)
    x0, y0 = int(tx - rot.shape[1] / 2), int(ty - rot.shape[0] / 2)
    _overlay_rgba(vis, rot, x0, y0)


# ─── PCA & Rotation ────────────────────────────────────────────────────────
#
# Frame: object X=right, Y=down, Z=toward camera (matches ArUco solvePnP).
# R: object-to-camera, p_cam = R @ p_obj. Columns of R = object axes in camera frame.

def orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho


def _pca_core(points: np.ndarray):
    """PCA: center, eigenvectors (cols), eigenvalues. vecs[:,2] = surface normal."""
    c = np.median(points, axis=0)
    X = points - c
    C = (X.T @ X) / max(1, X.shape[0] - 1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if np.linalg.det(vecs) < 0:
        vecs[:, 2] *= -1
    return c.astype(np.float64), vecs.astype(np.float64), vals.astype(np.float64)


def _align_pca_to_aruco_convention(vecs: np.ndarray) -> np.ndarray:
    """
    Map PCA eigenvectors to ArUco frame: X right, Y down, Z toward camera.
    PCA: col0,1 = in-plane, col2 = normal. ArUco: col0=X, col1=Y, col2=Z.
    """
    normal = vecs[:, 2]
    if normal[2] > 0:
        normal = -normal
    inplane0 = vecs[:, 0]
    if inplane0[0] < 0:
        inplane0 = -inplane0
    inplane1 = np.cross(normal, inplane0)
    inplane1 = inplane1 / max(1e-9, np.linalg.norm(inplane1))
    if inplane1[1] < 0:
        inplane1 = -inplane1
    inplane0 = np.cross(inplane1, normal)
    inplane0 = inplane0 / max(1e-9, np.linalg.norm(inplane0))
    R = np.column_stack([inplane0, inplane1, normal])
    return orthonormalize_rotation(R)


def pca_orientation(points: np.ndarray, refine: bool = True):
    if points.shape[0] < 150:
        raise ValueError("Not enough points for PCA")
    c, vecs, vals = _pca_core(points)
    X = points - c
    if refine and points.shape[0] > 300:
        lam1, lam2, lam3 = vals
        planarity = (lam2 - lam3) / lam1 if lam1 > 1e-12 else 0
        if planarity > 0.15:
            dists = np.abs(X @ vecs[:, 2])
            thr = np.clip(float(np.median(dists)) * 2.5, 0.002, 0.006)
            keep = dists < thr
            if np.sum(keep) >= 200:
                c, vecs, vals = _pca_core(points[keep])
    R = _align_pca_to_aruco_convention(vecs)
    return c, R, vals


def pca_orientation_ransac(points: np.ndarray):
    if points.shape[0] < 200:
        c, vecs, vals = _pca_core(points)
        return c, _align_pca_to_aruco_convention(vecs), vals
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.004, ransac_n=3, num_iterations=100,
        )
        if len(inliers) >= 150:
            pts_plane = points[np.asarray(inliers)]
            c, vecs, vals = _pca_core(pts_plane)
            return c, _align_pca_to_aruco_convention(vecs), vals
    except Exception:
        pass
    c, vecs, vals = _pca_core(points)
    return c, _align_pca_to_aruco_convention(vecs), vals


def generate_signed_permutations(R: np.ndarray):
    candidates = []
    for perm in itertools.permutations(range(3)):
        P = R[:, list(perm)].copy()
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            C = P.copy()
            C[:, 0] *= signs[0]
            C[:, 1] *= signs[1]
            C[:, 2] *= signs[2]
            if np.linalg.det(C) > 0:
                candidates.append(orthonormalize_rotation(C))
    return candidates


def rotation_geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic distance on SO(3) in degrees. Ra, Rb: object-to-camera."""
    val = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(val, -1.0, 1.0))))


def align_rotation_to_reference(R_raw: np.ndarray, R_ref: np.ndarray | None):
    R_raw = orthonormalize_rotation(R_raw)
    if R_ref is None:
        return R_raw, False, None
    R_ref = orthonormalize_rotation(R_ref)
    best_R, best_err = None, float("inf")
    for Rcand in generate_signed_permutations(R_raw):
        err = rotation_geodesic_deg(Rcand, R_ref)
        if err < best_err:
            best_err, best_R = err, Rcand
    return best_R, True, best_err


def _wrap_angle_rad(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def rotmat_to_rpy_zyx(R: np.ndarray):
    """
    Extract RPY from R (object-to-camera). ZYX Euler.
    R columns = (X, Y, Z) = (right, down, toward_camera) per ArUco.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    roll = _wrap_angle_rad(roll)
    pitch = _wrap_angle_rad(pitch)
    yaw = _wrap_angle_rad(yaw)
    if RPY_ROLL_NEGATE:
        roll = -roll
    return roll, pitch, yaw


def smooth_vec(prev, new, alpha=0.20):
    return (1.0 - alpha) * prev + alpha * new


def smooth_rotation(prev_R, new_R, alpha=0.20, max_jump_deg=45.0):
    gap = rotation_geodesic_deg(new_R, prev_R)
    if gap > max_jump_deg:
        return orthonormalize_rotation(new_R)
    return orthonormalize_rotation((1.0 - alpha) * prev_R + alpha * new_R)


def draw_axes(vis, K, dist, R_est, center_est, axis_len=0.04):
    """Draw axes: X=red, Y=green, Z=blue (ArUco: right, down, toward)."""
    center = np.asarray(center_est, dtype=np.float64).reshape(3)
    pts_3d = np.float32([
        center,
        center + axis_len * R_est[:, 0],
        center + axis_len * R_est[:, 1],
        center + axis_len * R_est[:, 2],
    ])
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    pts2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
    pts2d = pts2d.reshape(-1, 2)
    if pts2d.shape[0] != 4 or not np.all(np.isfinite(pts2d)):
        return
    o = tuple(np.round(pts2d[0]).astype(int))
    px = tuple(np.round(pts2d[1]).astype(int))
    py = tuple(np.round(pts2d[2]).astype(int))
    pz = tuple(np.round(pts2d[3]).astype(int))
    h, w = vis.shape[:2]
    max_len = max(w, h) * 0.5
    if (np.linalg.norm(np.array(px) - np.array(o)) > max_len or
            np.linalg.norm(np.array(py) - np.array(o)) > max_len or
            np.linalg.norm(np.array(pz) - np.array(o)) > max_len):
        return
    cv2.line(vis, o, px, (0, 0, 255), 2, cv2.LINE_AA)   # X = red  = roll
    cv2.line(vis, o, py, (0, 255, 0), 2, cv2.LINE_AA)   # Y = green = pitch
    cv2.line(vis, o, pz, (255, 0, 0), 2, cv2.LINE_AA)   # Z = blue  = yaw


def estimate_cube_center_from_markers(cube_markers, cube_size_m):
    if not cube_markers:
        return None
    half = float(cube_size_m) * 0.5
    shifted = []
    for m in cube_markers:
        t = m["tvec"].reshape(3).astype(np.float64)
        n = m["R"][:, 2].astype(np.float64)
        n = n / max(1e-9, np.linalg.norm(n))
        shifted.append([t + n * half, t - n * half])
    if len(shifted) == 1:
        c0, c1 = shifted[0]
        return c0 if np.linalg.norm(c0) < np.linalg.norm(c1) else c1
    best_score = float("inf")
    best_centers = None
    for choice in itertools.product([0, 1], repeat=len(shifted)):
        centers = np.array([shifted[i][choice[i]] for i in range(len(shifted))], dtype=np.float64)
        score = float(np.sum(np.linalg.norm(centers - centers.mean(axis=0), axis=1)))
        if score < best_score:
            best_score, best_centers = score, centers
    return np.mean(best_centers, axis=0) if best_centers is not None else None


def point_cloud_quality(depth_m, bbox, z_min=0.20, z_max=2.00):
    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = map(int, np.clip(bbox, [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1]))
    if x2 <= x1 or y2 <= y1:
        return 0.0, np.nan, np.nan
    roi = depth_m[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi) & (roi > z_min) & (roi < z_max)]
    if valid.size == 0:
        return 0.0, np.nan, np.nan
    return float(valid.size / max(1, roi.size)), float(np.median(valid)), float(np.std(valid))


def roi_depth_to_points(depth_m, intr, bbox, z_min=0.20, z_max=2.00, sample_step=2, band=0.025,
                        max_points=6000, trim_ratio=0.02, fg_percentile=15):
    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = np.clip([x1, x2], 0, w - 1)
    y1, y2 = np.clip([y1, y2], 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3), np.float32)
    trim_x = int((x2 - x1) * trim_ratio)
    trim_y = int((y2 - y1) * trim_ratio)
    x1t = min(max(x1 + trim_x, 0), w - 1)
    x2t = max(min(x2 - trim_x, w - 1), x1t + 1)
    y1t = min(max(y1 + trim_y, 0), h - 1)
    y2t = max(min(y2 - trim_y, h - 1), y1t + 1)
    roi = depth_m[y1t:y2t:sample_step, x1t:x2t:sample_step]
    if roi.size == 0:
        return np.empty((0, 3), np.float32)
    valid = roi[(roi > z_min) & (roi < z_max) & np.isfinite(roi)]
    if valid.size < 250:
        return np.empty((0, 3), np.float32)
    z_fg = float(np.percentile(valid, fg_percentile))
    mask = (roi > z_min) & (roi < z_max) & np.isfinite(roi) & (roi >= z_fg - 0.005) & (roi <= z_fg + band)
    ys, xs = np.where(mask)
    if ys.size < 150:
        z_med = float(np.median(valid))
        mask = (roi > z_min) & (roi < z_max) & np.isfinite(roi) & (np.abs(roi - z_med) < band)
        ys, xs = np.where(mask)
        if ys.size < 150:
            mask = (roi > z_min) & (roi < z_max) & np.isfinite(roi)
            ys, xs = np.where(mask)
            if ys.size < 150:
                return np.empty((0, 3), np.float32)
    us = xs * sample_step + x1t
    vs = ys * sample_step + y1t
    zs = depth_m[vs, us].astype(np.float32)
    if us.shape[0] > max_points:
        idx = np.linspace(0, us.shape[0] - 1, max_points, dtype=int)
        us, vs, zs = us[idx], vs[idx], zs[idx]
    pts = np.empty((us.shape[0], 3), np.float32)
    use_rs = hasattr(intr, "fx") and hasattr(rs, "rs2_deproject_pixel_to_point")
    if use_rs:
        try:
            for i, (u, v, z) in enumerate(zip(us, vs, zs)):
                pts[i] = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z))
        except Exception:
            use_rs = False
    if not use_rs:
        fx = getattr(intr, "fx", 582.6)
        fy = getattr(intr, "fy", 582.6)
        ppx = getattr(intr, "ppx", 313.0)
        ppy = getattr(intr, "ppy", 238.4)
        for i, (u, v, z) in enumerate(zip(us, vs, zs)):
            x = (float(u) - ppx) / fx * float(z)
            y = (float(v) - ppy) / fy * float(z)
            pts[i] = np.array([x, y, float(z)], dtype=np.float32)
    return pts


def estimate_obb_dimensions(points, center, R_est):
    local = (points.astype(np.float64) - center.reshape(1, 3)) @ R_est
    min_xyz = np.min(local, axis=0)
    max_xyz = np.max(local, axis=0)
    extents = max_xyz - min_xyz
    obb_center_local = (min_xyz + max_xyz) / 2.0
    obb_center = (obb_center_local @ R_est.T + center.reshape(1, 3)).flatten()
    sort_desc = np.sort(extents)[::-1]
    length_m, width_m, height_m = float(sort_desc[0]), float(sort_desc[1]), float(sort_desc[2])
    return length_m, width_m, height_m, (float(extents[0]), float(extents[1]), float(extents[2])), obb_center


def build_camera_matrix_and_dist(intr):
    K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)
    return K, dist


# ─── ArUco ─────────────────────────────────────────────────────────────────

def create_aruco_detector(dict_name=cv2.aruco.DICT_6X6_250):
    dictionary = cv2.aruco.getPredefinedDictionary(dict_name)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin, params.adaptiveThreshWinSizeMax = 3, 23
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate, params.maxMarkerPerimeterRate = 0.02, 4.0
    params.polygonalApproxAccuracyRate = 0.03
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_aruco_pose(gray, detector, K, dist, marker_length_m=0.04, edge_margin_px=12):
    markers = []
    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return markers
    h, w = gray.shape[:2]
    half = marker_length_m / 2.0
    objp = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    for corners, marker_id in zip(corners_list, ids.flatten()):
        imgp = corners.reshape(4, 2).astype(np.float32)
        if (np.any(imgp[:, 0] < edge_margin_px) or np.any(imgp[:, 0] > w - edge_margin_px) or
                np.any(imgp[:, 1] < edge_margin_px) or np.any(imgp[:, 1] > h - edge_margin_px)):
            continue
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue
        tvec = tvec.reshape(3, 1)
        z = float(tvec[2, 0])
        if z <= 0.02 or z > 2.50:
            continue
        Rm, _ = cv2.Rodrigues(rvec)
        markers.append({
            "id": int(marker_id),
            "corners": imgp,
            "rvec": rvec.reshape(3, 1),
            "tvec": tvec,
            "R": Rm.astype(np.float64),
            "center_px": (float(np.mean(imgp[:, 0])), float(np.mean(imgp[:, 1]))),
            "distance_m": float(np.linalg.norm(tvec)),
        })
    return markers


def _project_points(points_3d, rvec, tvec, K, dist):
    pts2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist)
    return pts2d.reshape(-1, 2)


def draw_marker_axis(img, K, dist, rvec, tvec, axis_len_m=0.02):
    axis_3d = np.float32([[0, 0, 0], [axis_len_m, 0, 0], [0, axis_len_m, 0], [0, 0, axis_len_m]])
    pts2d = _project_points(axis_3d, rvec, tvec, K, dist)
    if pts2d.shape[0] != 4 or not np.all(np.isfinite(pts2d)):
        return False
    h, w = img.shape[:2]
    max_len = max(w, h) * 0.7
    o, px, py, pz = [tuple(np.round(p).astype(int)) for p in pts2d]
    if (np.linalg.norm(np.array(px) - np.array(o)) > max_len or
            np.linalg.norm(np.array(py) - np.array(o)) > max_len or
            np.linalg.norm(np.array(pz) - np.array(o)) > max_len):
        return False
    cv2.line(img, o, px, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, o, py, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(img, o, pz, (255, 0, 0), 2, cv2.LINE_AA)
    return True


def pixel_to_ray(intr, u, v):
    d = np.array([(float(u) - intr.ppx) / intr.fx, (float(v) - intr.ppy) / intr.fy, 1.0], dtype=np.float64)
    return d / max(1e-12, np.linalg.norm(d))


def polygon_depth_points(depth_m, intr, polygon_xy, inner_scale=0.72, sample_step=2, z_min=0.15, z_max=2.0):
    corners = polygon_xy.astype(np.float32)
    center = corners.mean(axis=0)
    inner = center + (corners - center) * inner_scale
    h, w = depth_m.shape[:2]
    x1 = int(max(0, np.floor(np.min(inner[:, 0]))))
    x2 = int(min(w - 1, np.ceil(np.max(inner[:, 0]))))
    y1 = int(max(0, np.floor(np.min(inner[:, 1]))))
    y2 = int(min(h - 1, np.ceil(np.max(inner[:, 1]))))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3), np.float32)
    mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(inner - np.array([x1, y1])).astype(np.int32), 255)
    pts = []
    for v in range(y1, y2 + 1, sample_step):
        for u in range(x1, x2 + 1, sample_step):
            if mask[v - y1, u - x1] == 0:
                continue
            z = float(depth_m[v, u])
            if np.isfinite(z) and z_min < z < z_max:
                pts.append(rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], z))
    return np.asarray(pts, dtype=np.float32) if pts else np.empty((0, 3), np.float32)


def fit_plane_svd(points_xyz):
    if points_xyz.shape[0] < 30:
        return None, None
    center = points_xyz.mean(axis=0).astype(np.float64)
    _, _, vh = np.linalg.svd(points_xyz.astype(np.float64) - center, full_matrices=False)
    normal = vh[-1] / max(1e-12, np.linalg.norm(vh[-1]))
    if normal[2] > 0:
        normal = -normal
    return center, normal


def intersect_ray_plane(ray_dir, plane_point, plane_normal):
    denom = float(np.dot(plane_normal, ray_dir))
    if abs(denom) < 1e-9:
        return None
    t = float(np.dot(plane_normal, plane_point) / denom)
    return ray_dir * t if t > 0 else None


def measure_marker_from_depth(depth_m, intr, marker, z_min=0.15, z_max=2.0):
    corners = marker["corners"].astype(np.float32)
    plane_pts = polygon_depth_points(depth_m, intr, corners, inner_scale=0.72, sample_step=2, z_min=z_min, z_max=z_max)
    if plane_pts.shape[0] < 60:
        return None
    plane_center, plane_normal = fit_plane_svd(plane_pts)
    if plane_center is None:
        return None
    corners_3d = []
    for u, v in corners:
        ray = pixel_to_ray(intr, u, v)
        P = intersect_ray_plane(ray, plane_center, plane_normal)
        if P is None or not np.all(np.isfinite(P)):
            return None
        corners_3d.append(P)
    corners_3d = np.asarray(corners_3d, dtype=np.float64)
    e01 = np.linalg.norm(corners_3d[0] - corners_3d[1])
    e12 = np.linalg.norm(corners_3d[1] - corners_3d[2])
    e23 = np.linalg.norm(corners_3d[2] - corners_3d[3])
    e30 = np.linalg.norm(corners_3d[3] - corners_3d[0])
    width_m = 0.5 * (e01 + e23)
    height_m = 0.5 * (e12 + e30)
    return {
        "width_cm": width_m * 100.0,
        "height_cm": height_m * 100.0,
        "mean_edge_cm": 0.25 * (e01 + e12 + e23 + e30) * 100.0,
        "std_edge_cm": float(np.std([e01, e12, e23, e30])) * 100.0,
    }


def draw_aruco_markers(img, markers, K, dist, marker_axis_len_m=0.02, draw_axis=True, axis_exclude_ids=None):
    axis_exclude_ids = set() if axis_exclude_ids is None else {int(x) for x in axis_exclude_ids}
    for m in markers:
        corners = m["corners"].astype(int)
        for i in range(4):
            cv2.line(img, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2, cv2.LINE_AA)
        for p in corners:
            cv2.circle(img, tuple(p), 3, (0, 0, 255), -1)
        cx, cy = m["center_px"]
        cv2.putText(img, f"id={m['id']} d={m['distance_m']:.2f}m", (int(cx) - 30, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)
        if m.get("measured"):
            cv2.putText(img, f"meas={m['measured']['mean_edge_cm']:.2f}cm", (int(cx) - 42, int(cy) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        if draw_axis and int(m["id"]) not in axis_exclude_ids:
            draw_marker_axis(img, K, dist, m["rvec"], m["tvec"], axis_len_m=marker_axis_len_m)


def match_marker_to_bbox(markers, bbox, margin_px=30, max_3d_dist_m=0.40):
    if not markers:
        return None
    x1, y1, x2, y2 = map(int, bbox)
    bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    inside = [m for m in markers if (x1 - margin_px) <= m["center_px"][0] <= (x2 + margin_px) and
              (y1 - margin_px) <= m["center_px"][1] <= (y2 + margin_px)]
    if inside:
        inside.sort(key=lambda m: (m["center_px"][0] - bx) ** 2 + (m["center_px"][1] - by) ** 2)
        return inside[0]
    nearby = [m for m in markers if float(np.linalg.norm(m["tvec"].reshape(3))) < max_3d_dist_m]
    if not nearby:
        return None
    nearby.sort(key=lambda m: (m["center_px"][0] - bx) ** 2 + (m["center_px"][1] - by) ** 2)
    return nearby[0]


def bbox_iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter_area / max(1e-9, areaA + areaB - inter_area)


def markers_to_union_bbox(markers, img_shape, marker_ids, pad_px=18):
    selected = [m for m in markers if int(m["id"]) in marker_ids]
    if not selected:
        return None
    all_xy = np.concatenate([m["corners"].astype(np.float32) for m in selected], axis=0)
    h, w = img_shape[:2]
    x1 = max(0, int(np.floor(np.min(all_xy[:, 0])) - pad_px))
    y1 = max(0, int(np.floor(np.min(all_xy[:, 1])) - pad_px))
    x2 = min(w - 1, int(np.ceil(np.max(all_xy[:, 0])) + pad_px))
    y2 = min(h - 1, int(np.ceil(np.max(all_xy[:, 1])) + pad_px))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def markers_to_min_area_box(markers, marker_ids, scale=1.12):
    selected = [m for m in markers if int(m["id"]) in marker_ids]
    if not selected:
        return None
    all_xy = np.concatenate([m["corners"].astype(np.float32) for m in selected], axis=0)
    rect = cv2.minAreaRect(all_xy)
    box = cv2.boxPoints(rect).astype(np.float32)
    center = np.mean(box, axis=0, keepdims=True)
    return np.round((box - center) * float(scale) + center).astype(int)


def marker_to_bbox(marker, img_shape, pad_px=35):
    h, w = img_shape[:2]
    corners = marker["corners"].astype(np.float32)
    x1 = max(0, int(np.floor(np.min(corners[:, 0])) - pad_px))
    y1 = max(0, int(np.floor(np.min(corners[:, 1])) - pad_px))
    x2 = min(w - 1, int(np.ceil(np.max(corners[:, 0])) + pad_px))
    y2 = min(h - 1, int(np.ceil(np.max(corners[:, 1])) + pad_px))
    return [x1, y1, x2, y2]


def rotation_error_rad(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    """
    Geodesic distance on SO(3): angle of relative rotation R_gt @ R_est.T.
    Both R_gt, R_est: object-to-camera. Returns angle in radians [0, pi].
    """
    R_delta = R_gt @ R_est.T
    val = (np.trace(R_delta) - 1.0) / 2.0
    return math.acos(float(np.clip(val, -1.0, 1.0)))


def rotation_error_deg(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    return math.degrees(rotation_error_rad(R_gt, R_est))


def surface_normal_error_deg(R_gt, R_est):
    gt_normal = R_gt[:, 2]
    best_angle = 180.0
    for col in range(3):
        for sign in [1.0, -1.0]:
            cos_a = float(np.clip(np.dot(gt_normal, sign * R_est[:, col]), -1.0, 1.0))
            best_angle = min(best_angle, math.degrees(math.acos(cos_a)))
    return best_angle


def translation_surface_err(tvec, obb_center, R_est, extents):
    diff = tvec.reshape(3) - obb_center.reshape(3)
    err_total = float(np.linalg.norm(diff))
    min_idx = int(np.argmin(extents))
    err_normal = abs(float(np.dot(diff, R_est[:, min_idx])))
    err_tangent = float(np.sqrt(max(0.0, err_total ** 2 - err_normal ** 2)))
    return err_total, err_normal, err_tangent


# ─── Point cloud view ─────────────────────────────────────────────────────

_OBJ_COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (200, 0, 255), (0, 255, 200), (255, 255, 0)]


def render_pointcloud(depth_m, intr, detected_objs=None, canvas_h=480, canvas_w=640,
                      z_min=0.15, z_max=2.0, bg_step=6, pitch_deg=-25.0, yaw_deg=15.0):
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (15, 15, 15)
    h, w = depth_m.shape[:2]
    ys, xs = np.mgrid[0:h:bg_step, 0:w:bg_step]
    zs = depth_m[ys, xs].astype(np.float32)
    valid = np.isfinite(zs) & (zs > z_min) & (zs < z_max)
    xs_v, ys_v, zs_v = xs[valid].astype(np.float32), ys[valid].astype(np.float32), zs[valid]
    x3d = (xs_v - intr.ppx) / intr.fx * zs_v
    y3d = (ys_v - intr.ppy) / intr.fy * zs_v
    bg_pts = np.column_stack([x3d, y3d, zs_v])
    centroid = bg_pts.mean(axis=0) if bg_pts.shape[0] > 0 else np.array([0.0, 0.0, 1.0])
    a_p, a_y = np.radians(pitch_deg), np.radians(yaw_deg)
    Rp = np.array([[1, 0, 0], [0, np.cos(a_p), -np.sin(a_p)], [0, np.sin(a_p), np.cos(a_p)]], dtype=np.float64)
    Ry = np.array([[np.cos(a_y), 0, np.sin(a_y)], [0, 1, 0], [-np.sin(a_y), 0, np.cos(a_y)]], dtype=np.float64)
    R_view = Rp @ Ry
    scale = canvas_w * 0.45
    cx_c, cy_c = canvas_w / 2.0, canvas_h / 2.0

    def _project(pts3d):
        centered = pts3d - centroid.reshape(1, 3)
        rot = (R_view @ centered.T).T
        rot[:, 2] += centroid[2]
        px = (rot[:, 0] / rot[:, 2] * scale + cx_c).astype(int)
        py = (rot[:, 1] / rot[:, 2] * scale + cy_c).astype(int)
        depth_sort = rot[:, 2]
        ok = (px >= 0) & (px < canvas_w) & (py >= 0) & (py < canvas_h) & (depth_sort > 0.05)
        return px, py, depth_sort, ok

    colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, 256), cv2.COLORMAP_JET)[0]
    if bg_pts.shape[0] > 0:
        px, py, ds, ok = _project(bg_pts)
        d_norm = np.clip((ds[ok] - z_min) / (z_max - z_min), 0, 1)
        canvas[py[ok], px[ok]] = colormap[(d_norm * 255).astype(np.uint8)]
    if detected_objs:
        for idx, obj in enumerate(detected_objs):
            if obj["pts"].shape[0] < 10:
                continue
            color = _OBJ_COLORS[idx % len(_OBJ_COLORS)]
            px, py, _, ok = _project(obj["pts"])
            canvas[py[ok], px[ok]] = color
            if obj.get("label") and px[ok].shape[0] > 0:
                lx, ly = int(np.median(px[ok])), int(np.min(py[ok])) - 8
                cv2.putText(canvas, obj["label"], (lx, max(ly, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "3D Point Cloud", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    yolo_model_size, device = "nano", "cuda"
    conf_threshold, iou_threshold = 0.25, 0.45
    rs_w, rs_h, rs_fps = 640, 480, 15
    z_min, z_max = 0.20, 2.00
    sample_step, band = 1, 0.015
    marker_length_m = 0.04
    axis_len_m = 0.04
    pos_alpha, rot_alpha = 0.40, 0.15

    isaac_target_positions = {
        "cube_5cm": np.array([-0.3, -0.3, 0.12]),
        "cube_8cm": np.array([-0.3, 0.0, 0.12]),
        "cube_large": np.array([-0.3, 0.3, 0.12]),
        "default": np.array([-0.3, -0.3, 0.12]),
    }
    enable_cube_marker_detection = True
    cube_known_size_cm = 5.0
    cube_size_thresholds_cm = [6.0, 9.0]
    cube_size_labels = ["cube_5cm", "cube_8cm", "cube_large"]
    cube_class_id, cube_object_id = 999, 9001
    cube_class_name, cube_bbox_pad_px = "cube", 35
    cube_marker_ids = {11, 12}
    marker_axis_len_m = 0.02

    np.random.seed(0)
    init_pose_csv()
    dashboard.start(host="0.0.0.0", port=5000)

    detector = ObjectDetector(model_size=yolo_model_size, conf_thres=conf_threshold, iou_thres=iou_threshold, device=device)
    rs_cam = RealSenseDepth(w=rs_w, h=rs_h, fps=rs_fps)
    intr = rs_cam.intr
    K, dist = build_camera_matrix_and_dist(intr)
    aruco_detector = create_aruco_detector(cv2.aruco.DICT_6X6_250)
    isaac_sock = create_isaac_udp_client(host="127.0.0.1", port=6000)

    pose_state = {}
    frame_id = frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"

    print("Running: Pose + ArUco. Press P to save, Q/ESC to quit.")

    try:
        while True:
            color, depth_m, _ = rs_cam.read(timeout_ms=15000)
            if color is None or depth_m is None:
                continue

            vis = color.copy()
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            markers = detect_aruco_pose(gray, aruco_detector, K, dist, marker_length_m=marker_length_m, edge_margin_px=12)
            for m in markers:
                m["measured"] = measure_marker_from_depth(depth_m, intr, m, z_min=z_min, z_max=z_max)

            draw_aruco_markers(vis, markers, K, dist, marker_axis_len_m=marker_axis_len_m, draw_axis=True, axis_exclude_ids=cube_marker_ids)

            try:
                _, detections = detector.detect(color.copy(), track=True)
            except Exception as e:
                print("Detection error:", e)
                detections = []

            gt_only_markers = [m for m in markers if int(m["id"]) not in cube_marker_ids]
            if gt_only_markers:
                gt_bboxes = [marker_to_bbox(m, color.shape, pad_px=5) for m in gt_only_markers]
                detections = [d for d in detections if not any(bbox_iou(d[0], mb) > 0.3 for mb in gt_bboxes)]

            cube_display_box = None
            cube_markers = []
            if enable_cube_marker_detection:
                cube_markers = [m for m in markers if int(m["id"]) in cube_marker_ids]
                if cube_markers:
                    avg_edge_px = float(np.mean([np.linalg.norm(m["corners"][0] - m["corners"][1]) for m in cube_markers]))
                    adaptive_pad_px = int(avg_edge_px / marker_length_m * 0.008) + 3
                    cube_display_box = markers_to_min_area_box(markers, cube_marker_ids, scale=1.12)
                else:
                    adaptive_pad_px = cube_bbox_pad_px
                cube_bbox = markers_to_union_bbox(markers, color.shape, cube_marker_ids, pad_px=adaptive_pad_px)
                if cube_bbox is not None:
                    detections = [d for d in detections if bbox_iou(d[0], cube_bbox) < 0.35]
                    detections.append([cube_bbox, 1.0, cube_class_id, cube_object_id])

            rows_to_save = []
            pc_objs = []
            saved_pts = saved_center = saved_R = None

            for obj_index, det in enumerate(detections):
                bbox, score, class_id, obj_id = det
                cube_edge_est_cm = cube_size_cm = cube_size_label = ""

                if class_id == cube_class_id:
                    class_name = cube_class_name
                else:
                    names = detector.get_class_names()
                    class_name = names[class_id] if isinstance(names, (list, tuple)) and class_id < len(names) else str(class_id)

                valid_depth_ratio, z_med, z_std = point_cloud_quality(depth_m, bbox, z_min=z_min, z_max=z_max)
                pts = roi_depth_to_points(depth_m, intr, bbox, z_min=z_min, z_max=z_max, sample_step=sample_step, band=band, max_points=8000, trim_ratio=0.05)
                if pts.shape[0] < 250:
                    continue

                _bx1, _by1, _bx2, _by2 = bbox
                _cx_px, _cy_px = (_bx1 + _bx2) / 2.0, (_by1 + _by2) / 2.0
                _z_fg = float(np.percentile(pts[:, 2], 20))
                _obj_center_3d = np.array([(_cx_px - intr.ppx) / intr.fx * _z_fg, (_cy_px - intr.ppy) / intr.fy * _z_fg, _z_fg], dtype=np.float64)
                _obj_radius = math.sqrt((_bx2 - _bx1) ** 2 + (_by2 - _by1) ** 2) / intr.fx * _z_fg * 0.55
                pts = pts[np.linalg.norm(pts - _obj_center_3d, axis=1) < _obj_radius]
                if pts.shape[0] < 200:
                    continue

                _ctr = np.median(pts, axis=0)
                _dists2 = np.linalg.norm(pts - _ctr, axis=1)
                pts = pts[_dists2 < np.mean(_dists2) + 1.5 * np.std(_dists2)]
                if pts.shape[0] < 150:
                    continue

                try:
                    pca_fn = pca_orientation_ransac if PCA_USE_RANSAC else pca_orientation
                    center_pca, R_est_raw, eigvals = pca_fn(pts)
                    m = match_marker_to_bbox(markers, bbox)
                    x1, y1, x2, y2 = map(int, bbox)
                    marker_on_object = False
                    if m is not None:
                        mcx, mcy = m["center_px"]
                        marker_on_object = (x1 <= mcx <= x2) and (y1 <= mcy <= y2)

                    pose_key = f"{class_name}_{obj_id}" if obj_id is not None else f"{class_name}_single"
                    prev_R = pose_state[pose_key]["R"] if pose_key in pose_state else None
                    R_ref = m["R"] if (m and m.get("R") is not None) else prev_R
                    R_pca, perm_applied, perm_err_deg = align_rotation_to_reference(R_est_raw, R_ref)

                    use_marker_for_display = marker_on_object and m is not None and USE_MARKER_POSE_WHEN_ON_OBJECT
                    if use_marker_for_display:
                        R_est = m["R"].copy()
                        center_est = m["tvec"].reshape(3).astype(np.float64)
                        length_m, width_m, height_m, extents_alg, obb_center_alg = estimate_obb_dimensions(pts, center_pca, R_pca)
                        extents, obb_center = extents_alg, obb_center_alg
                        R_logged, center_logged = R_pca.copy(), obb_center_alg.copy()
                    else:
                        R_est = R_pca.copy()
                        center_est = center_pca.copy()
                        if pose_key in pose_state:
                            center_est = smooth_vec(pose_state[pose_key]["center"], center_est, alpha=pos_alpha)
                            R_est = smooth_rotation(pose_state[pose_key]["R"], R_est, alpha=rot_alpha)
                        length_m, width_m, height_m, extents, obb_center = estimate_obb_dimensions(pts, center_est, R_est)
                        R_logged, center_logged = R_est.copy(), obb_center.copy()
                    pose_state[pose_key] = {"center": center_est.copy(), "R": R_est.copy()}

                    roll_est, pitch_est, yaw_est = rotmat_to_rpy_zyx(R_est)

                    lam1, lam2, lam3 = [float(x) for x in eigvals]
                    linearity = (lam1 - lam2) / lam1 if lam1 > 1e-12 else np.nan
                    planarity = (lam2 - lam3) / lam1 if lam1 > 1e-12 else np.nan
                    scattering = lam3 / lam1 if lam1 > 1e-12 else np.nan
                except Exception:
                    continue

                gt_row = {}

                if m is not None:
                    tvec = m["tvec"].reshape(3)
                    R_gt = m["R"]
                    pos_err, err_normal, err_tangent = translation_surface_err(tvec, obb_center, R_logged, extents)
                    ang_err = rotation_error_deg(R_gt, R_logged) if marker_on_object else float("nan")
                    algorithm_ang_err = ang_err
                    normal_err = surface_normal_error_deg(R_gt, R_logged) if marker_on_object else float("nan")
                    gt_roll, gt_pitch, gt_yaw = rotmat_to_rpy_zyx(R_gt)
                    gt_row = {
                        "gt_marker_id": int(m["id"]),
                        "gt_distance_m": float(m["distance_m"]),
                        "gt_cx": float(tvec[0]), "gt_cy": float(tvec[1]), "gt_cz": float(tvec[2]),
                        "gt_roll_deg": math.degrees(gt_roll), "gt_pitch_deg": math.degrees(gt_pitch), "gt_yaw_deg": math.degrees(gt_yaw),
                        "gt_r00": float(R_gt[0, 0]), "gt_r01": float(R_gt[0, 1]), "gt_r02": float(R_gt[0, 2]),
                        "gt_r10": float(R_gt[1, 0]), "gt_r11": float(R_gt[1, 1]), "gt_r12": float(R_gt[1, 2]),
                        "gt_r20": float(R_gt[2, 0]), "gt_r21": float(R_gt[2, 1]), "gt_r22": float(R_gt[2, 2]),
                        "marker_on_object": marker_on_object,
                        "trans_err_m": pos_err, "trans_err_cm": pos_err * 100.0,
                        "dx": float(tvec[0] - obb_center[0]), "dy": float(tvec[1] - obb_center[1]), "dz": float(tvec[2] - obb_center[2]),
                        "trans_err_normal_cm": err_normal * 100.0, "trans_err_tangent_cm": err_tangent * 100.0,
                        "rot_err_deg": ang_err if not math.isnan(ang_err) else "",
                        "algorithm_rot_err_deg": algorithm_ang_err if not math.isnan(algorithm_ang_err) else "",
                        "normal_err_deg": normal_err if marker_on_object else "",
                    }
                    if m.get("measured"):
                        meas = m["measured"]
                        gt_row.update({
                            "marker_measured_width_cm": float(meas["width_cm"]),
                            "marker_measured_height_cm": float(meas["height_cm"]),
                            "marker_measured_mean_edge_cm": float(meas["mean_edge_cm"]),
                            "marker_measured_std_edge_cm": float(meas["std_edge_cm"]),
                            "marker_size_gt_cm": marker_length_m * 100.0,
                            "marker_size_err_cm": abs(float(meas["mean_edge_cm"]) - marker_length_m * 100.0),
                        })
                        if class_id == cube_class_id:
                            ce = float(meas["mean_edge_cm"])
                            cube_edge_est_cm = ce
                            cube_size_label = cube_size_labels[0] if ce < cube_size_thresholds_cm[0] else (cube_size_labels[1] if ce < cube_size_thresholds_cm[1] else cube_size_labels[2])

                if class_id == cube_class_id:
                    side_target_m = cube_known_size_cm / 100.0
                    length_m = width_m = height_m = side_target_m
                    extents = (side_target_m, side_target_m, side_target_m)
                    if cube_size_label:
                        class_name = cube_size_label
                    cube_size_cm = cube_known_size_cm

                axis_center = tvec if (marker_on_object and m is not None and USE_MARKER_POSE_WHEN_ON_OBJECT) else obb_center
                if class_id == cube_class_id and cube_markers:
                    cube_center = estimate_cube_center_from_markers(cube_markers, cube_known_size_cm / 100.0)
                    if cube_center is not None:
                        axis_center = cube_center
                        obb_center = cube_center
                        center_est = cube_center

                if class_id == cube_class_id and cube_display_box is not None:
                    cv2.polylines(vis, [cube_display_box.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_axes(vis, K, dist, R_est, axis_center, axis_len=axis_len_m)
                draw_text_along_edge(vis, (x1, y1), (x2, y1), f"L={length_m*100:.1f}cm", offset_px=-12)
                draw_text_along_edge(vis, (x2, y1), (x2, y2), f"H={height_m*100:.1f}cm", offset_px=8)
                draw_text_along_edge(vis, (x1, y2), (x2, y2), f"W={width_m*100:.1f}cm", offset_px=12)

                label = f"{class_name} {score:.2f}"
                if obj_id is not None:
                    label = f"ID:{obj_id} {label}"
                cv2.putText(vis, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

                y_txt = min(vis.shape[0] - 10, y2 + 18)
                rpy_txt = f"rpy=({math.degrees(roll_est):.1f},{math.degrees(pitch_est):.1f},{math.degrees(yaw_est):.1f})"
                cv2.putText(vis, rpy_txt, (x1, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, rpy_txt, (x1, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

                roll_logged, pitch_logged, yaw_logged = rotmat_to_rpy_zyx(R_logged)
                row = {
                    "frame": frame_id, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "object_index": int(obj_index), "object_id": int(obj_id) if obj_id is not None else "",
                    "class_id": int(class_id), "class_name": str(class_name), "confidence": float(score),
                    "bbox_x1": int(x1), "bbox_y1": int(y1), "bbox_x2": int(x2), "bbox_y2": int(y2),
                    "num_points": int(pts.shape[0]), "valid_depth_ratio": valid_depth_ratio,
                    "z_median_m": z_med, "z_std_m": z_std,
                    "cx": float(center_logged[0]), "cy": float(center_logged[1]), "cz": float(center_logged[2]),
                    "roll_deg": math.degrees(roll_logged), "pitch_deg": math.degrees(pitch_logged), "yaw_deg": math.degrees(yaw_logged),
                    "r00": float(R_logged[0, 0]), "r01": float(R_logged[0, 1]), "r02": float(R_logged[0, 2]),
                    "r10": float(R_logged[1, 0]), "r11": float(R_logged[1, 1]), "r12": float(R_logged[1, 2]),
                    "r20": float(R_logged[2, 0]), "r21": float(R_logged[2, 1]), "r22": float(R_logged[2, 2]),
                    "length_m": length_m, "width_m": width_m, "height_m": height_m,
                    "extent_axis0_m": extents[0], "extent_axis1_m": extents[1], "extent_axis2_m": extents[2],
                    "eigval1": lam1, "eigval2": lam2, "eigval3": lam3,
                    "linearity": linearity, "planarity": planarity, "scattering": scattering,
                    "cube_edge_est_cm": cube_edge_est_cm, "cube_size_cm": cube_size_cm, "cube_size_label": cube_size_label,
                }
                row.update(gt_row)
                rows_to_save.append(row)

                pc_objs.append({"pts": pts, "center": obb_center.copy(), "R": R_est.copy(), "extents": extents, "label": f"{class_name} {score:.2f}"})
                if saved_pts is None:
                    saved_pts, saved_center, saved_R = pts.copy(), center_logged.copy(), R_logged.copy()

                if class_id == cube_class_id:
                    target_pos = isaac_target_positions.get(str(cube_size_label), isaac_target_positions["default"])
                    send_cube_to_isaac(isaac_sock, frame_id, class_name, center_est, R_est, length_m, width_m, height_m,
                                       cube_size_label, cube_size_cm, target_position=target_pos)
                    dashboard.push_udp_log(f"{class_name} | pos=({center_est[0]:.2f},{center_est[1]:.2f},{center_est[2]:.2f})m | size={cube_size_cm}cm")
                    dashboard.set_isaac_connected(True)

            frame_count += 1
            frame_id += 1
            if frame_count % 10 == 0:
                fps_display = f"FPS: {frame_count / max(1e-6, time.time() - start_time):.1f}"

            cv2.putText(vis, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Pose + ArUco", vis)

            pc_view = render_pointcloud(depth_m, intr, detected_objs=pc_objs, z_min=z_min, z_max=z_max)
            cv2.imshow("Point Cloud", pc_view)

            depth_vis = np.clip(np.nan_to_num(depth_m, nan=0.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("Depth", depth_color)

            dashboard.push_frame("vis", vis)
            dashboard.push_frame("depth", depth_color)
            try:
                _fps_val = float(fps_display.split(": ")[-1])
            except (ValueError, IndexError):
                _fps_val = 0.0
            dashboard.push_detections(rows_to_save, _fps_val, frame_id)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                save_frame_bundle(frame_id, color, vis, depth_m, depth_color_bgr=depth_color, pc_view_bgr=pc_view)
                if saved_pts is not None:
                    save_ply_xyz(saved_pts, f"{OUTDIR}/clouds/{frame_id:06d}_roi_cloud.ply")
                    np.save(f"{OUTDIR}/clouds/{frame_id:06d}_roi_cloud.npy", saved_pts)
                    save_pca_axes_ply(saved_pts, saved_center, saved_R, axis_len=0.06, path=f"{OUTDIR}/clouds/{frame_id:06d}_roi_pca.ply")
                for row in rows_to_save:
                    append_pose_row(row)
                print(f"[Saved] {len(rows_to_save)} pose rows")
            elif key == ord('q') or key == 27:
                break
    finally:
        try:
            rs_cam.stop()
        except Exception:
            pass
        try:
            elapsed = max(1e-6, time.time() - start_time)
            fps_val = frame_count / elapsed
            fps_path = f"{OUTDIR}/pose/fps.txt"
            with open(fps_path, "w", encoding="utf-8") as f:
                f.write(f"{fps_val:.2f}\n")
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
