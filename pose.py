import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/pose/pose_log.csv"

df = pd.read_csv(path)

pose_cols = [
    "frame", "class_name", "cx", "cy", "cz",
    "roll_deg", "pitch_deg", "yaw_deg",
    "length_m", "width_m", "height_m",
    "trans_err_cm", "rot_err_deg",
]
available = [c for c in pose_cols if c in df.columns]

print(f"Pose log: {path}  ({len(df)} rows)")
print(df[available].tail(20).to_string(index=False))
