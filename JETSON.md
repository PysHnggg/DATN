# Jetson Compatibility Notes

This project runs on both Windows and NVIDIA Jetson. Jetson has different defaults that can cause errors.

## Differences (Windows vs Jetson)

| Aspect | Windows | Jetson |
|--------|---------|--------|
| Python | 3.10+ | 3.8 (JetPack) |
| PyTorch | 2.x | 1.12 (JetPack) |
| Display | Always available | Often headless (SSH) |
| Ultralytics | Often 8.3+ | May be 8.0.x by default |

## Fixes Applied

1. **Python 3.8**: Type hints use `Optional[X]` instead of `X | None`.
2. **Ultralytics**: Requires `>=8.3.0` for YOLOv11. Run `pip install "ultralytics>=8.3.0"`.
3. **Headless**: When `DISPLAY` is unset, the app skips `cv2.imshow()` and uses the web dashboard at http://0.0.0.0:5000.
4. **Qt/X11**: `QT_QPA_PLATFORM=offscreen` is set when headless to avoid "could not connect to display".
5. **CUDA OOM**: Detection falls back to CPU if GPU runs out of memory.
6. **Matplotlib**: Uses `Agg` backend in plot scripts for headless operation.

## Quick Start on Jetson

```bash
pip install -r requirements.txt
python3 run.py
# Open http://<jetson-ip>:5000 in a browser for the dashboard
```
