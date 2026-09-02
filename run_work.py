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
import argparse
import time
import math
import itertools
import cv2
import json
import socket
import numpy as np
import pyrealsense2 as rs
from detection_model import ObjectDetector, MultiObjectDetector
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

# Display options
SHOW_EDGE_LENGTHS = False  # user requested to hide edge-length labels
# For a cube seen from a single RGB-D view, full 3D orientation is ambiguous because
# the point cloud usually contains only one visible surface. Unless an ArUco marker is
# on the cube, draw/log a stable canonical cube orientation instead of a noisy PCA pose.
USE_CANONICAL_CUBE_ORIENTATION_WITHOUT_MARKER = True

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
    "captured",   # 1 = frame được nhấn P, dùng để lọc trong evaluate.py
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


_isaac_send_count = 0


def send_cube_to_isaac(sock, frame_id, class_name, center_est, R_est, length_m, width_m, height_m,
                       target_position=None):
    """Send detected cube pose + size to Isaac Sim for pick-and-place simulation."""
    global _isaac_send_count
    if sock is None:
        return
    if target_position is None:
        target_position = np.array([-0.3, -0.3, 0.12])
    msg = {
        "frame": int(frame_id),
        "name": str(class_name),
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
        _isaac_send_count += 1
        if _isaac_send_count == 1:
            peer = None
            try:
                peer = sock.getpeername()
            except Exception:
                pass
            print(f"[Isaac UDP] ✓ Gửi UDP cube đầu tiên tới {peer}: pos=({center_est[0]:.2f},{center_est[1]:.2f},{center_est[2]:.2f})m")
        elif _isaac_send_count % 60 == 0:
            print(f"[Isaac UDP] đã gửi {_isaac_send_count} message tới Isaac Sim")
    except Exception as e:
        print(f"[Isaac UDP] ✗ Send failed: {e}")


def save_frame_bundle(frame_id, color_bgr, vis_bgr, depth_m, depth_color_bgr=None, pc_view_bgr=None):
    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{frame_id:06d}_{ts}"
    cv2.imwrite(f"{OUTDIR}/frames/{prefix}_rgb.png", color_bgr)
    cv2.imwrite(f"{OUTDIR}/frames/{prefix}_vis.png", vis_bgr)
    np.save(f"{OUTDIR}/depth/{prefix}_depth.npy", depth_m)
    if depth_color_bgr is not None:
        cv2.imwrite(f"{OUTDIR}/depth/{prefix}_depth_color.png", depth_color_bgr)
    else:
        dc = colorize_depth_bgr(depth_m, z_min=0.20, z_max=2.0)
        cv2.imwrite(f"{OUTDIR}/depth/{prefix}_depth_color.png", dc)
    if pc_view_bgr is not None:
        cv2.imwrite(f"{OUTDIR}/frames/{prefix}_pointcloud.png", pc_view_bgr)
    return prefix
    
    
def save_all_windows(prefix, windows_dict):
    """Save all currently displayed windows to disk."""
    if not windows_dict:
        return
    for name, img in windows_dict.items():
        if img is None:
            continue
        safe = str(name).strip().lower().replace(" ", "_").replace("+", "plus")
        safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-"))
        if not safe:
            continue
        cv2.imwrite(f"{OUTDIR}/frames/{prefix}_{safe}.png", img)
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


def smooth_rotation(prev_R, new_R, alpha=0.20, max_jump_deg=35.0):
    gap = rotation_geodesic_deg(new_R, prev_R)
    if gap > max_jump_deg:
        # Reject abrupt frame-to-frame flips from PCA ambiguity/outliers.
        return orthonormalize_rotation(prev_R)
    return orthonormalize_rotation((1.0 - alpha) * prev_R + alpha * new_R)


def smooth_extents(prev_extents, new_extents, alpha=0.18, max_scale_change=1.25):
    prev = np.asarray(prev_extents, dtype=np.float64).reshape(3)
    new = np.asarray(new_extents, dtype=np.float64).reshape(3)
    prev = np.clip(prev, 1e-4, None)
    new = np.clip(new, 1e-4, None)
    ratio = new / prev
    ratio = np.clip(ratio, 1.0 / max_scale_change, max_scale_change)
    bounded = prev * ratio
    smoothed = (1.0 - alpha) * prev + alpha * bounded
    return tuple(float(x) for x in np.clip(smoothed, 0.005, 2.0))


def stabilize_points_for_viz(prev_pts, new_pts, alpha=0.35, max_points=2500):
    if prev_pts is None or len(prev_pts) < 100 or len(new_pts) < 100:
        return new_pts
    a = np.asarray(prev_pts, dtype=np.float32)
    b = np.asarray(new_pts, dtype=np.float32)
    n = min(a.shape[0], b.shape[0], int(max_points))
    if n < 100:
        return new_pts
    ia = np.linspace(0, a.shape[0] - 1, n, dtype=int)
    ib = np.linspace(0, b.shape[0] - 1, n, dtype=int)
    a_s = a[ia]
    b_s = b[ib]
    c_prev = np.median(a_s, axis=0)
    c_new = np.median(b_s, axis=0)
    if float(np.linalg.norm(c_new - c_prev)) > 0.12:
        return new_pts
    b_s = (1.0 - alpha) * a_s + alpha * b_s
    return b_s.astype(np.float32)


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


def draw_obb_2d(vis, K, dist, center, R, extents, color=(0, 255, 0), thickness=2):
    """
    Project the 3D OBB and draw a 2D oriented rectangle (4 corners) on the image.
    Uses cv2.minAreaRect over the projected 8 corners so the rectangle
    tightly encloses the 3D-box silhouette in image space.

    Args:
        center : (3,) OBB center in camera frame.
        R      : (3,3) object-to-camera rotation. Columns = OBB axes.
        extents: (3,) full lengths along R[:,0], R[:,1], R[:,2].
    Returns:
        box_2d : (4,2) int array of rectangle corners (or None on failure).
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    extents = np.asarray(extents, dtype=np.float64).reshape(3)
    half = extents / 2.0
    signs = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float64)
    local = signs * half.reshape(1, 3)
    corners_cam = (R @ local.T).T + center.reshape(1, 3)

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    pts2d, _ = cv2.projectPoints(corners_cam.astype(np.float64), rvec, tvec, K, dist)
    pts2d = pts2d.reshape(-1, 2)
    if pts2d.shape[0] != 8 or not np.all(np.isfinite(pts2d)):
        return None

    h, w = vis.shape[:2]
    max_span = max(w, h) * 1.5
    c2d = np.mean(pts2d, axis=0)
    if np.any(np.linalg.norm(pts2d - c2d, axis=1) > max_span):
        return None

    rect = cv2.minAreaRect(pts2d.astype(np.float32))
    box = cv2.boxPoints(rect)
    box_int = np.round(box).astype(int)
    for i in range(4):
        cv2.line(vis, tuple(box_int[i]), tuple(box_int[(i + 1) % 4]),
                 color, thickness, cv2.LINE_AA)
    return box_int


def draw_obb_3d(vis, K, dist, center, R, extents, color=(0, 255, 255), thickness=2):
    """
    Draw full 3D OBB wireframe (8 projected corners + 12 edges).
    Returns projected 2D points (8,2) or None.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    extents = np.asarray(extents, dtype=np.float64).reshape(3)
    half = extents / 2.0
    signs = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float64)
    local = signs * half.reshape(1, 3)
    corners_cam = (R @ local.T).T + center.reshape(1, 3)

    if np.any(corners_cam[:, 2] <= 0.03) or not np.all(np.isfinite(corners_cam)):
        return None

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    pts2d, _ = cv2.projectPoints(corners_cam.astype(np.float64), rvec, tvec, K, dist)
    pts2d = pts2d.reshape(-1, 2)
    if pts2d.shape[0] != 8 or not np.all(np.isfinite(pts2d)):
        return None

    h, w = vis.shape[:2]
    max_span = max(w, h) * 1.8
    c2d = np.mean(pts2d, axis=0)
    if np.any(np.linalg.norm(pts2d - c2d, axis=1) > max_span):
        return None

    p = np.round(pts2d).astype(int)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # near face
        (4, 5), (5, 6), (6, 7), (7, 4),  # far face
        (0, 4), (1, 5), (2, 6), (3, 7),  # connectors
    ]
    for i, j in edges:
        cv2.line(vis, tuple(p[i]), tuple(p[j]), color, thickness, cv2.LINE_AA)
    return p


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
                        max_points=6000, trim_ratio=0.02, fg_percentile=15, max_thick_override=None):
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
    # Foreground extraction by depth histogram peak (closest cluster).
    # The extracted segment stops when we hit an EMPTY GAP in the histogram,
    # which is what physically separates the object from the table behind it.
    bin_w = 0.005
    z_lo_hist = float(np.percentile(valid, 1))
    z_hi_hist = float(np.percentile(valid, 99))
    if z_hi_hist - z_lo_hist < bin_w * 2:
        z_hi_hist = z_lo_hist + bin_w * 2
    nb = max(4, int(math.ceil((z_hi_hist - z_lo_hist) / bin_w)))
    hist, edges = np.histogram(valid, bins=nb, range=(z_lo_hist, z_hi_hist))
    if hist.max() <= 0:
        return np.empty((0, 3), np.float32)

    # Walk from front (closest) and pick the first significant peak.
    front_thr = max(30, int(0.20 * hist.max()))
    peak_idx = None
    for i in range(len(hist)):
        if hist[i] >= front_thr:
            peak_idx = i
            break
    if peak_idx is None:
        peak_idx = int(np.argmax(hist))

    # Expand around peak using a gap-based rule. We keep adding bins as long
    # as we have not seen 2 consecutive empty bins (>= 1cm gap).
    gap_bins = max(2, int(round(0.010 / bin_w)))
    empty_thr = max(2, int(0.02 * hist.max()))
    lo_i = peak_idx
    empties = 0
    while lo_i - 1 >= 0:
        if hist[lo_i - 1] <= empty_thr:
            empties += 1
            if empties >= gap_bins:
                break
        else:
            empties = 0
        lo_i -= 1
    hi_i = peak_idx
    empties = 0
    while hi_i + 1 < len(hist):
        if hist[hi_i + 1] <= empty_thr:
            empties += 1
            if empties >= gap_bins:
                break
        else:
            empties = 0
        hi_i += 1

    z_lo = float(edges[max(0, lo_i)]) - 0.003
    z_hi = float(edges[min(len(edges) - 1, hi_i + 1)]) + 0.003
    # Adaptive thickness cap: scale with bbox size so that tilted elongated
    # objects (e.g. remote at an angle) do not get truncated.
    if max_thick_override is not None:
        max_thickness = float(max_thick_override)
    else:
        max_thickness = float(np.clip(max(band, 0.10), 0.06, 0.20))
    if z_hi - z_lo > max_thickness:
        z_hi = z_lo + max_thickness
    mask = (roi > z_min) & (roi < z_max) & np.isfinite(roi) & (roi >= z_lo) & (roi <= z_hi)
    ys, xs = np.where(mask)
    if ys.size < 150:
        z_med = float(np.median(valid))
        mask = (roi > z_min) & (roi < z_max) & np.isfinite(roi) & (np.abs(roi - z_med) < max(0.03, band))
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


def remove_table_plane_ransac(points, distance_threshold=0.006, keep_margin_m=0.008,
                              min_points_after=120, prefer_small_side=True,
                              reference_point=None):
    """
    Remove the dominant table plane from a ROI point cloud.

    RealSense/OpenCV camera frame:
      X right, Y down, Z forward.

    The function fits a plane with Open3D RANSAC, removes inlier plane points,
    then keeps only one side of the plane. If reference_point is given, the side
    containing that point is preferred. Otherwise, the smaller side is assumed
    to be the object when table points dominate the ROI.
    """
    if points is None or points.shape[0] < 300:
        return points
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=float(distance_threshold),
            ransac_n=3,
            num_iterations=180,
        )
        if len(inliers) < max(80, int(0.12 * points.shape[0])):
            return points

        a, b, c, d = [float(x) for x in plane_model]
        n = np.array([a, b, c], dtype=np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            return points
        n /= n_norm
        d /= n_norm

        signed = points.astype(np.float64) @ n + d
        pos = signed > float(keep_margin_m)
        neg = signed < -float(keep_margin_m)

        if reference_point is not None and np.all(np.isfinite(reference_point)):
            ref_signed = float(np.asarray(reference_point, dtype=np.float64).reshape(3) @ n + d)
            obj_mask = pos if ref_signed >= 0 else neg
        elif prefer_small_side:
            # In most YOLO ROIs the table/background side has more points than the object.
            obj_mask = pos if int(pos.sum()) <= int(neg.sum()) else neg
        else:
            # Fallback: remove plane inliers only, keep both non-plane sides.
            obj_mask = np.abs(signed) > float(keep_margin_m)

        pts_obj = points[obj_mask]
        if pts_obj.shape[0] < int(min_points_after):
            non_plane = np.abs(signed) > float(distance_threshold)
            pts_obj = points[non_plane]
        if pts_obj.shape[0] < int(min_points_after):
            return points
        return pts_obj.astype(np.float32)
    except Exception as e:
        print("[WARN] remove_table_plane_ransac failed:", e)
        return points


def remove_pointcloud_outliers(points, nb_neighbors=18, std_ratio=1.8, min_points_after=120):
    """Light Open3D statistical outlier removal for noisy RealSense depth."""
    if points is None or points.shape[0] < 300:
        return points
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=int(nb_neighbors),
            std_ratio=float(std_ratio),
        )
        if len(ind) >= int(min_points_after):
            return points[np.asarray(ind, dtype=np.int64)].astype(np.float32)
    except Exception as e:
        print("[WARN] remove_pointcloud_outliers failed:", e)
    return points


def cube_center_from_marker(marker, cube_size_m, pts=None):
    """
    Estimate cube center from a marker mounted on one cube face.
    The marker pose tvec is on the marker plane; shift by half cube size along
    marker normal. Try both normal directions and choose the one closer to the
    object point cloud median if points are available.
    """
    if marker is None:
        return None
    t = marker["tvec"].reshape(3).astype(np.float64)
    n = marker["R"][:, 2].astype(np.float64)
    n /= max(1e-9, np.linalg.norm(n))
    c1 = t + n * (float(cube_size_m) * 0.5)
    c2 = t - n * (float(cube_size_m) * 0.5)
    if pts is not None and pts.shape[0] > 0:
        med = np.median(pts.astype(np.float64), axis=0)
        # Prefer candidate closer to cloud median, but avoid a center that moves
        # toward camera from the marker plane (usually wrong sign on cube depth axis).
        s1 = np.linalg.norm(c1 - med) + (40.0 if c1[2] < t[2] else 0.0)
        s2 = np.linalg.norm(c2 - med) + (40.0 if c2[2] < t[2] else 0.0)
        return c1 if s1 <= s2 else c2
    return c1 if np.linalg.norm(c1) <= np.linalg.norm(c2) else c2


def project_obb_bbox_2d(K, dist, center, R, extents):
    """Project 8 corners of a 3D OBB and return its 2D axis-aligned bbox."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    R = orthonormalize_rotation(np.asarray(R, dtype=np.float64).reshape(3, 3))
    extents = np.asarray(extents, dtype=np.float64).reshape(3)
    half = extents / 2.0
    signs = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float64)
    local = signs * half.reshape(1, 3)
    corners_cam = (R @ local.T).T + center.reshape(1, 3)
    if np.any(corners_cam[:, 2] <= 0.03) or not np.all(np.isfinite(corners_cam)):
        return None
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    pts2d, _ = cv2.projectPoints(corners_cam, rvec, tvec, K, dist)
    pts2d = pts2d.reshape(-1, 2)
    if not np.all(np.isfinite(pts2d)):
        return None
    return [float(np.min(pts2d[:, 0])), float(np.min(pts2d[:, 1])),
            float(np.max(pts2d[:, 0])), float(np.max(pts2d[:, 1]))]


def bbox_iou_float(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(x) for x in box_a]
    bx1, by1, bx2, by2 = [float(x) for x in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(1e-9, area_a + area_b - inter))


def bbox_center_size_error(proj_box, yolo_box):
    if proj_box is None:
        return 1e9
    px1, py1, px2, py2 = [float(x) for x in proj_box]
    yx1, yy1, yx2, yy2 = [float(x) for x in yolo_box]
    pcx, pcy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
    ycx, ycy = 0.5 * (yx1 + yx2), 0.5 * (yy1 + yy2)
    pw, ph = max(1.0, px2 - px1), max(1.0, py2 - py1)
    yw, yh = max(1.0, yx2 - yx1), max(1.0, yy2 - yy1)
    center_err = math.hypot(pcx - ycx, pcy - ycy)
    size_err = abs(math.log(pw / yw)) * 30.0 + abs(math.log(ph / yh)) * 30.0
    iou_bonus = bbox_iou_float(proj_box, yolo_box) * 80.0
    return center_err + size_err - iou_bonus


def estimate_cube_center_from_visible_surface(surface_center, R_est, cube_size_m, bbox, K, dist, pts=None):
    """
    A depth camera usually sees only the visible surface of a 5 cm cube, not the
    real cube center. Shift the visible-surface center by half the cube size along
    possible PCA axes and choose the shift whose projected 3D box best matches
    the YOLO bbox.
    """
    surface_center = np.asarray(surface_center, dtype=np.float64).reshape(3)
    R_est = orthonormalize_rotation(np.asarray(R_est, dtype=np.float64).reshape(3, 3))
    half = 0.5 * float(cube_size_m)
    extents = np.array([cube_size_m, cube_size_m, cube_size_m], dtype=np.float64)

    candidate_dirs = []
    for i in range(3):
        axis = R_est[:, i].astype(np.float64)
        axis /= max(1e-9, np.linalg.norm(axis))
        candidate_dirs.extend([axis, -axis])

    # Also try camera-Z shifts. This is useful when PCA on a small cube becomes unstable.
    candidate_dirs.extend([
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
    ])

    # If the point cloud is available, use its median as visible-surface center;
    # it is usually less biased by a few outliers than the PCA center.
    if pts is not None and pts.shape[0] > 0:
        surface_center = np.median(pts.astype(np.float64), axis=0)

    best_center = surface_center.copy()
    best_score = 1e18
    for n in candidate_dirs:
        c = surface_center + n * half
        proj_box = project_obb_bbox_2d(K, dist, c, R_est, extents)
        score = bbox_center_size_error(proj_box, bbox)
        # Penalize moving the cube center in front of the visible surface.
        if c[2] < surface_center[2]:
            score += 25.0
        # Strongly penalize shifting toward +blue axis (R[:,2]); for a visible
        # front surface, real cube center should lie on the negative blue side.
        shift_blue = float(np.dot(c - surface_center, R_est[:, 2]))
        if shift_blue > 0.0:
            score += 90.0
        if score < best_score:
            best_score = score
            best_center = c
    return best_center.astype(np.float64)


def estimate_obb_dimensions(points, center, R_est, q_low=2.0, q_high=98.0):
    # R_est is object-to-camera (columns = object axes in camera frame).
    # To project points into object frame we need R_est.T (camera-to-object).
    local = (points.astype(np.float64) - center.reshape(1, 3)) @ R_est.T
    # Robust extents to avoid OBB jitter from depth outliers.
    min_xyz = np.percentile(local, q_low, axis=0)
    max_xyz = np.percentile(local, q_high, axis=0)
    extents = max_xyz - min_xyz
    obb_center_local = (min_xyz + max_xyz) / 2.0
    # Transform center back to camera frame: @ R_est (not R_est.T)
    obb_center = (obb_center_local @ R_est + center.reshape(1, 3)).flatten()
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


def deduplicate_detections(detections, iou_threshold=0.5):
    """Loại bỏ detection trùng lặp: 1 vật chỉ còn 1 ID. Giữ box có confidence cao hơn."""
    if len(detections) < 2:
        return detections
    kept = []
    for det in sorted(detections, key=lambda d: -d[1]):  # sort by confidence descending
        bbox, score, class_id, obj_id = det
        overlaps = any(bbox_iou(bbox, k[0]) > iou_threshold for k in kept)
        if not overlaps:
            kept.append(det)
    return kept


def suppress_by_priority_class(detections, priority_class_id, iou_threshold=0.3):
    """
    Drop non-priority detections that overlap (IoU > threshold) with any priority
    detection. Handy when a specialized model (e.g. best.pt: cube) and a generic
    model (e.g. yolo11n.pt: sports ball / book / box) fire on the same object.
    """
    if priority_class_id is None or priority_class_id < 0 or len(detections) < 2:
        return detections
    priority = [d for d in detections if int(d[2]) == int(priority_class_id)]
    if not priority:
        return detections
    kept = list(priority)
    for d in detections:
        if int(d[2]) == int(priority_class_id):
            continue
        if any(bbox_iou(d[0], p[0]) > iou_threshold for p in priority):
            continue
        kept.append(d)
    return kept


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
_CUBE_PC_COLOR = (255, 0, 0)  # blue in BGR


def sample_point_colors_from_rgb(points_xyz, intr, color_bgr):
    """
    Sample BGR colors for 3D points by projecting them back to the RGB frame.
    Invalid projections use white fallback for better visibility.
    """
    if points_xyz is None or points_xyz.shape[0] == 0:
        return np.empty((0, 3), dtype=np.uint8)
    if color_bgr is None or color_bgr.size == 0:
        return np.full((points_xyz.shape[0], 3), 255, dtype=np.uint8)

    h, w = color_bgr.shape[:2]
    z = points_xyz[:, 2]
    valid_z = np.isfinite(z) & (z > 1e-6)

    colors = np.full((points_xyz.shape[0], 3), 255, dtype=np.uint8)
    if not np.any(valid_z):
        return colors

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    u = (x / z * intr.fx + intr.ppx).astype(np.int32)
    v = (y / z * intr.fy + intr.ppy).astype(np.int32)

    in_img = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if np.any(in_img):
        colors[in_img] = color_bgr[v[in_img], u[in_img]]
    return colors


def render_pointcloud(depth_m, intr, detected_objs=None, canvas_h=480, canvas_w=640,
                      z_min=0.15, z_max=2.0, bg_step=6, pitch_deg=-25.0, yaw_deg=15.0,
                      draw_frames=False, frame_axis_len=0.05, draw_obb_2d=False):
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
            px, py, _, ok = _project(obj["pts"])
            if bool(obj.get("is_cube", False)):
                label_color = _CUBE_PC_COLOR
                canvas[py[ok], px[ok]] = label_color
                if np.any(ok):
                    # Make cube points more visible than background samples.
                    px_ok, py_ok = px[ok], py[ok]
                    canvas[np.clip(py_ok - 1, 0, canvas_h - 1), px_ok] = label_color
                    canvas[np.clip(py_ok + 1, 0, canvas_h - 1), px_ok] = label_color
                    canvas[py_ok, np.clip(px_ok - 1, 0, canvas_w - 1)] = label_color
                    canvas[py_ok, np.clip(px_ok + 1, 0, canvas_w - 1)] = label_color
            else:
                point_colors = obj.get("point_colors")
                if isinstance(point_colors, np.ndarray) and point_colors.shape[0] == obj["pts"].shape[0]:
                    canvas[py[ok], px[ok]] = point_colors[ok]
                    label_color = tuple(int(c) for c in np.median(point_colors[ok], axis=0)) if np.any(ok) else (255, 255, 255)
                else:
                    label_color = _OBJ_COLORS[idx % len(_OBJ_COLORS)]
                    canvas[py[ok], px[ok]] = label_color
            if obj.get("label") and px[ok].shape[0] > 0:
                lx, ly = int(np.median(px[ok])), int(np.min(py[ok])) - 8
                cv2.putText(canvas, obj["label"], (lx, max(ly, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1, cv2.LINE_AA)
            if draw_obb_2d and np.count_nonzero(ok) >= 12:
                pts2 = np.column_stack([px[ok], py[ok]]).astype(np.float32)
                rect = cv2.minAreaRect(pts2)
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.polylines(canvas, [box], True, (0, 255, 255), 2, cv2.LINE_AA)
            if draw_frames and ("center" in obj) and ("R" in obj):
                try:
                    c = np.asarray(obj["center"], dtype=np.float64).reshape(3)
                    Rf = orthonormalize_rotation(np.asarray(obj["R"], dtype=np.float64).reshape(3, 3))
                    axis_pts = np.vstack([
                        c,
                        c + frame_axis_len * Rf[:, 0],
                        c + frame_axis_len * Rf[:, 1],
                        c + frame_axis_len * Rf[:, 2],
                    ])
                    ax_px, ax_py, _, ax_ok = _project(axis_pts)
                    if np.all(ax_ok):
                        o = (int(ax_px[0]), int(ax_py[0]))
                        px_x = (int(ax_px[1]), int(ax_py[1]))
                        px_y = (int(ax_px[2]), int(ax_py[2]))
                        px_z = (int(ax_px[3]), int(ax_py[3]))
                        cv2.line(canvas, o, px_x, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.line(canvas, o, px_y, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.line(canvas, o, px_z, (255, 0, 0), 2, cv2.LINE_AA)
                except Exception:
                    pass
    cv2.putText(canvas, "3D Point Cloud", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def fill_depth_holes_for_viz(depth_m, z_min=0.15, z_max=2.0):
    """
    Fill sparse/invalid depth holes for visualization only.
    This keeps algorithmic depth unchanged while improving Point Cloud display.
    """
    d = depth_m.astype(np.float32).copy()
    valid = np.isfinite(d) & (d > z_min) & (d < z_max)
    if np.count_nonzero(valid) < 100:
        return d

    # Iterative weighted box fill: invalid pixels borrow nearby valid depth.
    for k in (5, 9):
        inv = ~valid
        if not np.any(inv):
            break
        sum_d = cv2.blur(np.where(valid, d, 0.0), (k, k))
        sum_w = cv2.blur(valid.astype(np.float32), (k, k))
        fillable = inv & (sum_w > 1e-6)
        d[fillable] = sum_d[fillable] / sum_w[fillable]
        valid = np.isfinite(d) & (d > z_min) & (d < z_max)
    return d


def colorize_depth_bgr(depth_m, z_min=0.20, z_max=2.0, adaptive=True, p_lo=3.0, p_hi=97.0):
    """
    JET visualization: fixed scale hides detail when the whole scene sits in a
    narrow band (e.g. 0.45–0.6 m → almost one shade of blue with depth/2).
    When adaptive=True, stretch contrast using percentiles of valid pixels only.
    """
    d = np.nan_to_num(np.asarray(depth_m, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.isfinite(depth_m) & (d > z_min) & (d < z_max)
    out = np.zeros_like(d, dtype=np.float32)
    if adaptive and np.count_nonzero(valid) > 80:
        lo = float(np.percentile(d[valid], p_lo))
        hi = float(np.percentile(d[valid], p_hi))
        if hi - lo < 0.03:
            lo, hi = z_min, z_max
        span = max(hi - lo, 1e-6)
        out[valid] = np.clip((d[valid] - lo) / span, 0.0, 1.0)
    else:
        out[valid] = np.clip((d[valid] - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0)
    u8 = (out * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    _ap = argparse.ArgumentParser(description="YOLO-3D pose (RealSense + YOLO + PCA)")
    _ap.add_argument(
        "--profile-stages",
        action="store_true",
        help="Print algorithm-stage runtime (ms): 2D detection, ROI point cloud, PCA+OBB",
    )
    _args, _ = _ap.parse_known_args()
    profile_stages = _args.profile_stages

    yolo_weights_list = ["best.pt",]
    device = "cuda"
    conf_threshold, iou_threshold = 0.25, 0.45
    rs_w, rs_h, rs_fps = 640, 480, 15
    z_min, z_max = 0.20, 2.00
    sample_step, band = 1, 0.04
    marker_length_m = 0.04
    axis_len_m = 0.04
    pos_alpha, rot_alpha = 0.25, 0.06

    # Đo cạnh cube thật rồi sửa dòng này. Ví dụ 0.05 = 5 cm.
    # Với cube nhỏ + mặt bàn đen, không nên lấy size cube hoàn toàn từ depth vì depth hay bị thủng.
    cube_size_m = 0.05

    # Tách mặt bàn khỏi ROI point cloud trước khi PCA/OBB.
    REMOVE_TABLE_PLANE = True
    TABLE_RANSAC_DIST_M = 0.006
    TABLE_KEEP_MARGIN_M = 0.008

    # Single pick-and-place target position in Isaac Sim (robot base frame).
    isaac_target_position = np.array([-0.3, -0.3, 0.12])
    cube_class_name = "cube"
    # ArUco marker IDs physically mounted on the cube (GT only, NOT used to define cube)
    cube_marker_ids = {11, 12}
    marker_axis_len_m = 0.02

    # Deterministic: cùng 1 frame capture 2 lần ra cùng kết quả
    np.random.seed(0)
    try:
        import torch
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        import open3d as o3d
        if hasattr(o3d.utility, "random_seed"):
            o3d.utility.random_seed(0)
    except Exception:
        pass

    init_pose_csv()
    dashboard.start(host="0.0.0.0", port=5000)

    detector = MultiObjectDetector(
        weights_list=yolo_weights_list,
        conf_thres=conf_threshold, iou_thres=iou_threshold, device=device,
    )
    _det_names = detector.get_class_names()
    cube_class_id = next(
        (int(k) for k, v in _det_names.items() if str(v).lower() == cube_class_name),
        -1,
    )
    if cube_class_id < 0:
        print(f"[WARN] class '{cube_class_name}' not found in any model; cube-specific logic will be disabled.")
    else:
        print(f"[INFO] Cube class resolved to global_id={cube_class_id} ('{_det_names[cube_class_id]}')")
    print(f"[INFO] Loaded {len(yolo_weights_list)} detectors: {yolo_weights_list}")
    print(f"[INFO] Total classes across all models: {len(_det_names)}")
    rs_cam = RealSenseDepth(w=rs_w, h=rs_h, fps=rs_fps)
    intr = rs_cam.intr
    K, dist = build_camera_matrix_and_dist(intr)
    aruco_detector = create_aruco_detector(cv2.aruco.DICT_6X6_250)
    _isaac_host = os.environ.get("ISAAC_UDP_HOST", "127.0.0.1").strip() or "127.0.0.1"
    isaac_sock = create_isaac_udp_client(host=_isaac_host, port=6000)
    dashboard.configure_isaac_commands(host=_isaac_host, port=6001)
    dashboard.configure_cube_signal(host=_isaac_host, port=6000)
    print(f"[INFO] Isaac manual trigger → {_isaac_host}:6000  commands → {_isaac_host}:6001 (set ISAAC_UDP_HOST if Isaac runs on another PC)")

    pose_state = {}
    box_state = {}
    pc_viz_state = {}
    frame_id = frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"

    print("Running: Pose + ArUco. Press P to save, Q/ESC to quit.")
    if profile_stages:
        print("[profile] algorithm stages (ms): 2d_detection | roi_point_cloud | pca_pose_and_obb")

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

            _t_2d0 = time.perf_counter()
            try:
                _, detections = detector.detect(color.copy(), track=True)
            except Exception as e:
                print("Detection error:", e)
                detections = []
            _t_2d1 = time.perf_counter()
            detections_raw = list(detections)

            yolo_2d_raw_vis = color.copy()
            for det in detections_raw:
                bbox, score, class_id, obj_id = det
                bx1, by1, bx2, by2 = map(int, bbox)
                name_map = detector.get_class_names()
                if isinstance(name_map, dict):
                    cls_name = str(name_map.get(int(class_id), class_id))
                elif isinstance(name_map, (list, tuple)) and int(class_id) < len(name_map):
                    cls_name = str(name_map[int(class_id)])
                else:
                    cls_name = str(class_id)
                cls_label = f"{cls_name} {float(score):.2f}"
                if obj_id is not None:
                    cls_label = f"ID:{obj_id} {cls_label}"
                cv2.rectangle(yolo_2d_raw_vis, (bx1, by1), (bx2, by2), (0, 200, 255), 2)
                cv2.putText(yolo_2d_raw_vis, cls_label, (bx1, max(18, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(yolo_2d_raw_vis, cls_label, (bx1, max(18, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)

            # Filter out YOLO detections overlapping with GT-only markers (not the ones on the cube).
            # ArUco 11/12 on the cube are GT for the cube itself, not for other objects.
            gt_only_markers = [m for m in markers if int(m["id"]) not in cube_marker_ids]
            if gt_only_markers:
                gt_bboxes = [marker_to_bbox(m, color.shape, pad_px=5) for m in gt_only_markers]
                detections = [d for d in detections if not any(bbox_iou(d[0], mb) > 0.3 for mb in gt_bboxes)]

            # Prefer cube (best.pt) over generic COCO classes when they overlap on the same object.
            detections = suppress_by_priority_class(detections, cube_class_id, iou_threshold=0.3)
            # 1 vật chỉ 1 ID: loại detection chồng lấp (IoU > 0.5), giữ box confidence cao hơn
            detections = deduplicate_detections(detections, iou_threshold=0.5)
            yolo_2d_vis = color.copy()
            yolo_obb_vis = color.copy()
            for det in detections:
                bbox, score, class_id, obj_id = det
                bx1, by1, bx2, by2 = map(int, bbox)
                name_map = detector.get_class_names()
                if isinstance(name_map, dict):
                    cls_name = str(name_map.get(int(class_id), class_id))
                elif isinstance(name_map, (list, tuple)) and int(class_id) < len(name_map):
                    cls_name = str(name_map[int(class_id)])
                else:
                    cls_name = str(class_id)
                cls_label = f"{cls_name} {float(score):.2f}"
                if obj_id is not None:
                    cls_label = f"ID:{obj_id} {cls_label}"
                cv2.rectangle(yolo_2d_vis, (bx1, by1), (bx2, by2), (255, 180, 0), 2)
                cv2.putText(yolo_2d_vis, cls_label, (bx1, max(18, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(yolo_2d_vis, cls_label, (bx1, max(18, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA)
                cv2.rectangle(yolo_obb_vis, (bx1, by1), (bx2, by2), (255, 180, 0), 2)
                cv2.putText(yolo_obb_vis, cls_label, (bx1, max(18, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(yolo_obb_vis, cls_label, (bx1, max(18, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA)

            rows_to_save = []
            pc_objs = []
            pc_objs_before_geom = []
            pc_objs_after_geom = []
            pc_objs_ransac_before_pca = []
            pc_objs_after_pca = []
            pc_objs_final_with_frame = []
            saved_pts = saved_center = saved_R = None

            for obj_index, det in enumerate(detections):
                bbox, score, class_id, obj_id = det

                names = detector.get_class_names()
                if isinstance(names, dict):
                    class_name = str(names.get(int(class_id), class_id))
                elif isinstance(names, (list, tuple)) and class_id < len(names):
                    class_name = str(names[class_id])
                else:
                    class_name = str(class_id)
                is_cube = (int(class_id) == cube_class_id) or (class_name.lower() == cube_class_name)

                _t_dq0 = time.perf_counter()
                valid_depth_ratio, z_med, z_std = point_cloud_quality(depth_m, bbox, z_min=z_min, z_max=z_max)
                _t_dq1 = time.perf_counter()
                _bx1, _by1, _bx2, _by2 = bbox
                _cx_px, _cy_px = (_bx1 + _bx2) / 2.0, (_by1 + _by2) / 2.0
                _bw_px = max(1.0, _bx2 - _bx1)
                _bh_px = max(1.0, _by2 - _by1)

                # Bbox-adaptive depth thickness: scale with bbox shorter side.
                # Using the SHORTER side avoids over-clamping cubes that have
                # similar X/Y/Z extents (e.g. a 5cm cube needs ~6cm thickness,
                # not just 4cm derived from a thin remote bbox).
                _z_probe = float(np.nanmedian(np.where(np.isfinite(depth_m) & (depth_m > z_min) & (depth_m < z_max), depth_m, np.nan)))
                _bbox_short_3d = float(min(_bw_px, _bh_px) / float(intr.fx) * float(_z_probe if np.isfinite(_z_probe) else 0.5))
                _adaptive_thick = float(np.clip(_bbox_short_3d * 1.4, 0.06, 0.20))

                pts = roi_depth_to_points(
                    depth_m, intr, bbox,
                    z_min=z_min, z_max=z_max, sample_step=sample_step, band=band,
                    max_points=10000, trim_ratio=0.02,
                    max_thick_override=max(_adaptive_thick, cube_size_m * 1.8 if is_cube else _adaptive_thick),
                )
                if pts.shape[0] < 250:
                    continue

                # Match marker early so table removal can keep the side that contains the cube marker.
                m_pre = match_marker_to_bbox(markers, bbox)
                marker_on_object_pre = False
                if m_pre is not None:
                    _mx, _my = m_pre["center_px"]
                    marker_on_object_pre = (_bx1 <= _mx <= _bx2) and (_by1 <= _my <= _by2)
                ref_for_table = None
                if marker_on_object_pre and int(m_pre["id"]) in cube_marker_ids:
                    ref_for_table = m_pre["tvec"].reshape(3).astype(np.float64)

                if REMOVE_TABLE_PLANE:
                    _n_before_table = pts.shape[0]
                    _pts_pre_table = pts.copy()
                    pts_tr = remove_table_plane_ransac(
                        pts,
                        distance_threshold=TABLE_RANSAC_DIST_M,
                        keep_margin_m=TABLE_KEEP_MARGIN_M,
                        min_points_after=150,
                        prefer_small_side=True,
                        reference_point=ref_for_table,
                    )
                    pts_tr = remove_pointcloud_outliers(pts_tr, nb_neighbors=18, std_ratio=1.8, min_points_after=150)
                    # Fallback: if table removal eats too many points, keep the
                    # pre-removal cloud so OBB / 3D box can still be drawn.
                    if pts_tr is None or pts_tr.shape[0] < 180:
                        pts = _pts_pre_table
                        if frame_id % 15 == 0:
                            print(f"[table remove] frame={frame_id} obj={obj_index} {class_name}: {_n_before_table} -> {0 if pts_tr is None else pts_tr.shape[0]} (fallback to pre-removal)")
                    else:
                        pts = pts_tr
                        if frame_id % 15 == 0:
                            print(f"[table remove] frame={frame_id} obj={obj_index} {class_name}: {_n_before_table} -> {pts.shape[0]}")
                pts_before_geom = pts.copy()
                if pts_before_geom.shape[0] >= 120:
                    pc_objs_before_geom.append({
                        "pts": pts_before_geom,
                        "point_colors": sample_point_colors_from_rgb(pts_before_geom, intr, color),
                        "is_cube": bool(is_cube),
                        "label": f"{class_name} {score:.2f}",
                    })
                    pc_objs_ransac_before_pca.append({
                        "pts": pts_before_geom,
                        "point_colors": sample_point_colors_from_rgb(pts_before_geom, intr, color),
                        "is_cube": bool(is_cube),
                        "label": f"{class_name} {score:.2f}",
                    })

                # Inner-bbox projection filter: keep points projecting inside
                # the inner ~88% of the YOLO bbox to drop only the bbox border
                # bleed (e.g. thin strip of table). Lighter than a 70% cut so
                # we do not lose elongated objects like a remote.
                _shrink = 0.02 if is_cube else 0.06
                _ix1 = _cx_px - _bw_px * (0.5 - _shrink)
                _ix2 = _cx_px + _bw_px * (0.5 - _shrink)
                _iy1 = _cy_px - _bh_px * (0.5 - _shrink)
                _iy2 = _cy_px + _bh_px * (0.5 - _shrink)
                _u = pts[:, 0] / np.clip(pts[:, 2], 1e-3, None) * intr.fx + intr.ppx
                _v = pts[:, 1] / np.clip(pts[:, 2], 1e-3, None) * intr.fy + intr.ppy
                _in_inner = (_u >= _ix1) & (_u <= _ix2) & (_v >= _iy1) & (_v <= _iy2)
                if int(_in_inner.sum()) >= 250:
                    pts = pts[_in_inner]

                _z_fg = float(np.percentile(pts[:, 2], 20))
                _obj_center_3d = np.array([(_cx_px - intr.ppx) / intr.fx * _z_fg, (_cy_px - intr.ppy) / intr.fy * _z_fg, _z_fg], dtype=np.float64)
                _obj_radius = math.hypot(_bw_px, _bh_px) / float(intr.fx) * _z_fg * 0.55
                pts = pts[np.linalg.norm(pts - _obj_center_3d, axis=1) < _obj_radius]
                if pts.shape[0] < 200:
                    continue

                # Front-Z trim: useful for thin objects, but for cube it can cut away the rear/top face.
                if not is_cube:
                    _z_front = float(np.percentile(pts[:, 2], 5))
                    pts = pts[pts[:, 2] <= _z_front + _adaptive_thick * 1.2]
                    if pts.shape[0] < 200:
                        continue

                _ctr = np.median(pts, axis=0)
                _dists2 = np.linalg.norm(pts - _ctr, axis=1)
                pts = pts[_dists2 < np.mean(_dists2) + 1.5 * np.std(_dists2)]
                if pts.shape[0] < 150:
                    continue

                # Plane-slab refinement: fit a quick PCA, keep only points
                # within a slab around the dominant surface normal. Slab is
                # adaptive so curved/protruding object features (buttons on
                # remote, etc.) are not over-trimmed.
                try:
                    _c0 = np.median(pts, axis=0)
                    _X0 = pts.astype(np.float64) - _c0
                    _C0 = (_X0.T @ _X0) / max(1, _X0.shape[0] - 1)
                    _vals0, _vecs0 = np.linalg.eigh(_C0)
                    _normal0 = _vecs0[:, np.argmin(_vals0)]
                    if _normal0[2] > 0:
                        _normal0 = -_normal0
                    _d0 = (pts - _c0) @ _normal0
                    _d_med = float(np.median(_d0))
                    _slab = float(np.clip(np.std(_d0) * 2.5, 0.020, 0.060))
                    _slab_mask = np.abs(_d0 - _d_med) < _slab
                    if int(_slab_mask.sum()) >= 250:
                        pts = pts[_slab_mask]
                except Exception:
                    pass

                pts_after_geom = pts.copy()
                if pts_after_geom.shape[0] >= 120:
                    _colors_after_geom = sample_point_colors_from_rgb(pts_after_geom, intr, color)
                    pc_objs_after_geom.append({
                        "pts": pts_after_geom,
                        "point_colors": _colors_after_geom,
                        "is_cube": bool(is_cube),
                        "label": f"{class_name} {score:.2f}",
                    })

                _t_pc1 = time.perf_counter()

                try:
                    _t_pca0 = time.perf_counter()
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

                    # In-plane outlier trim using PCA local frame (MAD-based).
                    # Removes the strip of background/table points along the
                    # bbox edge that share Z with the object surface.
                    try:
                        _local = (pts.astype(np.float64) - center_pca.reshape(1, 3)) @ R_pca
                        _med = np.median(_local, axis=0)
                        _mad = np.median(np.abs(_local - _med), axis=0) + 1e-9
                        _keep = (
                            (np.abs(_local[:, 0] - _med[0]) < 3.5 * _mad[0]) &
                            (np.abs(_local[:, 1] - _med[1]) < 3.5 * _mad[1]) &
                            (np.abs(_local[:, 2] - _med[2]) < 4.0 * _mad[2])
                        )
                        if int(_keep.sum()) >= 200:
                            pts = pts[_keep]
                            center_pca, R_est_raw, eigvals = pca_fn(pts)
                            R_pca, perm_applied, perm_err_deg = align_rotation_to_reference(R_est_raw, R_ref)
                    except Exception:
                        pass

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
                    # For a known cube, use the measured physical size for display/Isaac/CSV size.
                    # Depth on dark tables often sees only a partial cube surface, so percentile extents
                    # will underestimate the true box.
                    if is_cube:
                        # Cube thật là 5 cm. Do not estimate cube size from partial/noisy depth.
                        length_m = width_m = height_m = float(cube_size_m)
                        extents = (float(cube_size_m), float(cube_size_m), float(cube_size_m))

                        # For a small cube, the depth cloud is usually only the visible surface.
                        # Therefore center_est/center_pca is a surface center, not the real cube center.
                        c_marker = cube_center_from_marker(m, cube_size_m, pts) if (marker_on_object and m is not None) else None
                        if c_marker is not None:
                            obb_center = c_marker.astype(np.float64)
                        else:
                            obb_center = estimate_cube_center_from_visible_surface(
                                surface_center=center_est,
                                R_est=R_est,
                                cube_size_m=cube_size_m,
                                bbox=bbox,
                                K=K,
                                dist=dist,
                                pts=pts,
                            )
                        center_est = obb_center.copy()

                        # Draw/log cube box with the displayed pose. If marker is on the cube, R_est is marker pose.
                        # Without marker, a cube's orientation from partial depth is unstable, so use a
                        # stable canonical orientation for the displayed/logged OBB.
                        if USE_CANONICAL_CUBE_ORIENTATION_WITHOUT_MARKER and not (marker_on_object and m is not None and USE_MARKER_POSE_WHEN_ON_OBJECT):
                            R_est = np.eye(3, dtype=np.float64)
                        R_logged = R_est.copy()
                        center_logged = obb_center.copy()


                    # Additional temporal smoothing for displayed OBB center/size.
                    # Pose smoothing above stabilizes R/center_est; this block
                    # stabilizes OBB extents + OBB center specifically.
                    if pose_key in box_state:
                        prev_box = box_state[pose_key]
                        obb_center = smooth_vec(prev_box["center"], obb_center, alpha=0.22)
                        extents = smooth_extents(prev_box["extents"], extents, alpha=0.18, max_scale_change=1.25)
                        length_m, width_m, height_m = sorted([float(extents[0]), float(extents[1]), float(extents[2])], reverse=True)
                        center_logged = obb_center.copy()
                    box_state[pose_key] = {
                        "center": np.asarray(obb_center, dtype=np.float64).copy(),
                        "extents": np.asarray(extents, dtype=np.float64).copy(),
                    }

                    pose_state[pose_key] = {"center": center_est.copy(), "R": R_est.copy()}

                    _t_pca1 = time.perf_counter()
                    if profile_stages:
                        ms_roi = (_t_pc1 - _t_dq0) * 1000.0
                        ms_pose = (_t_pca1 - _t_pca0) * 1000.0
                        print(
                            f"[profile] f={frame_id} obj={obj_index} {class_name}: "
                            f"roi_point_cloud={ms_roi:.2f} pca_pose_and_obb={ms_pose:.2f}"
                        )

                    pc_objs_after_pca.append({
                        "pts": pts_after_geom,
                        "point_colors": sample_point_colors_from_rgb(pts_after_geom, intr, color),
                        "is_cube": bool(is_cube),
                        "center": center_logged.copy(),
                        "R": R_logged.copy(),
                        "label": f"{class_name} {score:.2f}",
                    })
                    pc_objs_final_with_frame.append({
                        "pts": pts_after_geom,
                        "point_colors": sample_point_colors_from_rgb(pts_after_geom, intr, color),
                        "is_cube": bool(is_cube),
                        "center": center_logged.copy(),
                        "R": R_logged.copy(),
                        "label": f"{class_name} {score:.2f}",
                    })

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
                axis_center = tvec if (marker_on_object and m is not None and USE_MARKER_POSE_WHEN_ON_OBJECT) else obb_center

                obb_corners_2d = draw_obb_2d(
                    vis, K, dist, obb_center, R_logged, extents,
                    color=(0, 255, 255), thickness=1,
                )
                draw_obb_2d(
                    yolo_obb_vis, K, dist, obb_center, R_logged, extents,
                    color=(0, 255, 255), thickness=2,
                )
                draw_obb_3d(
                    vis, K, dist, obb_center, R_logged, extents,
                    color=(255, 255, 0), thickness=2,
                )
                # Always show YOLO 2D bbox as the primary box on Pose + ArUco view.
                # OBB (if available) is drawn in yellow only as pose/size reference.
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_axes(vis, K, dist, R_est, axis_center, axis_len=axis_len_m)
                draw_axes(yolo_obb_vis, K, dist, R_logged, obb_center, axis_len=axis_len_m)

                if SHOW_EDGE_LENGTHS:
                    if obb_corners_2d is not None:
                        # 2D OBB has 4 corners; pick the longer edge as L, the
                        # shorter (perpendicular) as W. H is shown as a separate text.
                        e_a = float(np.linalg.norm(obb_corners_2d[1] - obb_corners_2d[0]))
                        e_b = float(np.linalg.norm(obb_corners_2d[2] - obb_corners_2d[1]))
                        if e_a >= e_b:
                            long_edge, short_edge = (0, 1), (1, 2)
                        else:
                            long_edge, short_edge = (1, 2), (0, 1)
                        p1 = tuple(obb_corners_2d[long_edge[0]])
                        p2 = tuple(obb_corners_2d[long_edge[1]])
                        draw_text_along_edge(vis, p1, p2, f"L={length_m*100:.1f}cm", offset_px=-10)
                        p1 = tuple(obb_corners_2d[short_edge[0]])
                        p2 = tuple(obb_corners_2d[short_edge[1]])
                        draw_text_along_edge(vis, p1, p2, f"W={width_m*100:.1f}cm", offset_px=-10)
                        cx_box = int(np.mean(obb_corners_2d[:, 0]))
                        top_y = int(np.min(obb_corners_2d[:, 1]) - 8)
                        h_text = f"H={height_m*100:.1f}cm"
                        cv2.putText(vis, h_text, (cx_box - 40, max(top_y, 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(vis, h_text, (cx_box - 40, max(top_y, 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    else:
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
                }
                row.update(gt_row)
                rows_to_save.append(row)

                _prev_viz_pts = pc_viz_state.get(pose_key)
                _viz_pts = stabilize_points_for_viz(_prev_viz_pts, pts, alpha=0.35, max_points=2500)
                pc_viz_state[pose_key] = _viz_pts.copy()
                _viz_colors = sample_point_colors_from_rgb(_viz_pts, intr, color)
                pc_objs.append({
                    "pts": _viz_pts,
                    "point_colors": _viz_colors,
                    "is_cube": bool(is_cube),
                    "center": obb_center.copy(),
                    "R": R_logged.copy(),
                    "extents": extents,
                    "label": f"{class_name} {score:.2f}",
                })
                if saved_pts is None:
                    saved_pts, saved_center, saved_R = pts.copy(), center_logged.copy(), R_logged.copy()

                if is_cube:
                    # Manual-trigger flow:
                    #   YOLO detects cube -> dashboard shows detection -> user presses
                    #   "Send Cube Signal" -> dashboard sends UDP trigger to Isaac.
                    #
                    # Do NOT auto-send pose/size/rotation to Isaac here.
                    # Isaac will spawn/reset the demo cube using its own fixed sample values.
                    dashboard.set_status("udp", True, "cube detected - press Send Cube Signal")
                    if frame_id % 30 == 0:
                        dashboard.push_log(
                            "detector",
                            f"Cube detected ({class_name}, conf={score:.2f}). Press 'Send Cube Signal' on dashboard.",
                            level="ok",
                        )

            if profile_stages:
                ms_2d = (_t_2d1 - _t_2d0) * 1000.0
                print(f"[profile] f={frame_id} 2d_detection={ms_2d:.1f}ms")

            frame_count += 1
            frame_id += 1
            if frame_count % 10 == 0:
                fps_display = f"FPS: {frame_count / max(1e-6, time.time() - start_time):.1f}"

            cv2.putText(vis, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Pose + ArUco", vis)
            cv2.imshow("RGB Raw", color)
            cv2.imshow("RGB + YOLO 2D Raw", yolo_2d_raw_vis)
            cv2.imshow("RGB + YOLO 2D Final", yolo_2d_vis)
            cv2.imshow("RGB + YOLO + 2D OBB + PCA Axis", yolo_obb_vis)

            # Visualization-only: fill holes + adaptive depth coloring (algorithms still use raw depth_m).
            depth_pc_viz = fill_depth_holes_for_viz(depth_m, z_min=z_min, z_max=z_max)
            # Main view (objects highlighted).
            pc_view = render_pointcloud(depth_pc_viz, intr, detected_objs=pc_objs, z_min=z_min, z_max=z_max, bg_step=4)
            cv2.imshow("Point Cloud", pc_view)
            # Raw view: denser sampling to avoid sparse/patchy appearance.
            pc_raw_view = render_pointcloud(depth_pc_viz, intr, detected_objs=None, z_min=z_min, z_max=z_max, bg_step=2)
            cv2.imshow("Point Cloud Raw", pc_raw_view)
            pc_before_geom = render_pointcloud(depth_pc_viz, intr, detected_objs=pc_objs_before_geom, z_min=z_min, z_max=z_max, bg_step=4)
            cv2.imshow("Object Foreground Cloud Before Geometric Filtering", pc_before_geom)
            pc_after_geom = render_pointcloud(depth_pc_viz, intr, detected_objs=pc_objs_after_geom, z_min=z_min, z_max=z_max, bg_step=4)
            cv2.imshow("Cloud After Geometric Filtering", pc_after_geom)
            pc_ransac_before_pca = render_pointcloud(depth_pc_viz, intr, detected_objs=pc_objs_ransac_before_pca, z_min=z_min, z_max=z_max, bg_step=4)
            cv2.imshow("RANSAC Before PCA", pc_ransac_before_pca)
            pc_after_pca = render_pointcloud(depth_pc_viz, intr, detected_objs=pc_objs_after_pca, z_min=z_min, z_max=z_max, bg_step=4)
            cv2.imshow("After PCA", pc_after_pca)
            pc_final_with_frame = render_pointcloud(
                depth_pc_viz, intr, detected_objs=pc_objs_final_with_frame, z_min=z_min, z_max=z_max, bg_step=4,
                draw_frames=True, frame_axis_len=axis_len_m, draw_obb_2d=True,
            )
            cv2.imshow("Final Point Cloud with Estimated Coordinate Frame", pc_final_with_frame)

            depth_color = colorize_depth_bgr(depth_m, z_min=z_min, z_max=z_max)
            cv2.imshow("Depth", depth_color)

            dashboard.push_frame("vis", vis)
            dashboard.push_frame("depth", depth_color)
            try:
                _fps_val = float(fps_display.split(": ")[-1])
            except (ValueError, IndexError):
                _fps_val = 0.0
            dashboard.push_detections(rows_to_save, _fps_val, frame_id)

            key = cv2.waitKey(1)
            key_code = key & 0xFF
            if key_code in (ord('p'), ord('P')):
                prefix = save_frame_bundle(frame_id, color, vis, depth_m, depth_color_bgr=depth_color, pc_view_bgr=pc_view)
                save_all_windows(prefix, {
                    "pose_aruco": vis,
                    "rgb_raw": color,
                    "rgb_yolo_2d_raw": yolo_2d_raw_vis,
                    "rgb_yolo_2d_final": yolo_2d_vis,
                    "rgb_yolo_2d_obb_pca_axis": yolo_obb_vis,
                    "point_cloud": pc_view,
                    "point_cloud_raw": pc_raw_view,
                    "object_foreground_cloud_before_geometric_filtering": pc_before_geom,
                    "cloud_after_geometric_filtering": pc_after_geom,
                    "ransac_before_pca": pc_ransac_before_pca,
                    "after_pca": pc_after_pca,
                    "final_point_cloud_with_estimated_coordinate_frame": pc_final_with_frame,
                    "depth": depth_color,
                })
                if saved_pts is not None:
                    save_ply_xyz(saved_pts, f"{OUTDIR}/clouds/{frame_id:06d}_roi_cloud.ply")
                    np.save(f"{OUTDIR}/clouds/{frame_id:06d}_roi_cloud.npy", saved_pts)
                    save_pca_axes_ply(saved_pts, saved_center, saved_R, axis_len=0.06, path=f"{OUTDIR}/clouds/{frame_id:06d}_roi_pca.ply")
                for row in rows_to_save:
                    row["captured"] = 1
                    append_pose_row(row)
                print(f"[Saved] frame={frame_id}  {len(rows_to_save)} detection(s)")
            elif key_code in (ord('q'), ord('Q'), 27):
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