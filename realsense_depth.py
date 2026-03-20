import numpy as np
import pyrealsense2 as rs
import cv2

class RealSenseDepth:
    def __init__(self, w=640, h=480, fps=15):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(cfg)

        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        # warmup
        for _ in range(20):
            self.pipeline.wait_for_frames(15000)

        frames = self.align.process(self.pipeline.wait_for_frames(15000))
        depth = frames.get_depth_frame()
        self.intr = depth.profile.as_video_stream_profile().intrinsics

    def read(self, timeout_ms=15000):
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms))
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return None, None, None

        depth_u16 = np.asanyarray(depth.get_data())              # uint16
        depth_m = depth_u16.astype(np.float32) * self.depth_scale # meters
        color_bgr = np.asanyarray(color.get_data())              # BGR
        return color_bgr, depth_m, depth_u16

    def colorize_depth(self, depth_m):
        # normalize for visualization only
        dm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.clip(dm, 0.0, 2.0)  # show 0..2m
        v = (v / 2.0 * 255).astype(np.uint8)
        return cv2.applyColorMap(v, cv2.COLORMAP_JET)

    def stop(self):
        self.pipeline.stop()