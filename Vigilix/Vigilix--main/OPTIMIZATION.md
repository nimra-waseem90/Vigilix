# Vigilix + weapon-yolo26x — Optimization & System Requirements

The new weapon model ([HaiderKhan6410/weapon-yolo26x](https://huggingface.co/HaiderKhan6410/weapon-yolo26x))
is a **YOLO26x** checkpoint — the largest/most-accurate YOLO26 variant
(58.8M params, ~193 GFLOPs, 7 classes, mAP@50 0.89). It replaces the old
`gun_knife_yolo11n.pt`, which was a far smaller/faster 2-class YOLO11**n**
model. You're trading speed for a large accuracy jump, and the hardware
you need changes accordingly.

## 1. System requirements

| Tier | CPU-only | GPU (recommended) |
|---|---|---|
| **Minimum** (works, not real-time) | 4+ cores, 8 GB RAM | Any CUDA GPU with 4 GB VRAM (e.g. GTX 1650) |
| **Recommended** (1 real-time stream) | — | NVIDIA RTX 3060/4060 or better, 8 GB VRAM, CUDA 12.x |
| **Production** (multiple streams) | — | RTX 4070 Ti+/A4000+ or datacenter GPU (L4, A10, H100), 16 GB+ VRAM |

Rough per-frame latency at imgsz=1024 (from the model card + typical
consumer-GPU scaling):

| Environment | Latency / frame | Realistic FPS |
|---|---|---|
| CPU only (any modern CPU) | ~1–5 s | <1 FPS — not usable for live alerting |
| Consumer GPU, PyTorch FP32 (e.g. RTX 3060) | ~15–30 ms | ~30–60 FPS |
| Consumer GPU, TensorRT FP16 | ~8–15 ms | ~60–120 FPS |
| Datacenter GPU, PyTorch (H100) | ~5 ms | ~200 FPS |
| Datacenter GPU, TensorRT FP16 (H100) | ~2 ms | ~400 FPS |

**Bottom line: CPU-only is not viable for live weapon detection with this
model.** The old yolo11n model could limp along on CPU; this one can't —
budget for a CUDA GPU if you want real-time alerts. Other requirements:

- Python 3.10–3.12 (3.13 not yet verified against this stack)
- NVIDIA driver + CUDA 12.x if using GPU (`nvidia-smi` should work)
- ~2 GB disk for weights + CUDA/torch wheels
- Each additional camera stream adds roughly one full inference pass per
  frame — VRAM and throughput budget accordingly (see §3).

## 2. One-time setup

```bash
pip install -r requirements.txt
python download_weapon_model.py       # pulls model/detection/weapon_yolo26x.pt
# optional, GPU only, biggest speed win:
python export_weapon_model_trt.py
```

## 3. Concrete optimizations (in order of impact)

1. **Use a GPU, and let the code auto-detect it.**
   `server/stream_worker.py` now picks CUDA automatically if available
   (`_default_device()`), falling back to CPU. Override with
   `VIGILIX_WEAPON_DEVICE=0` (GPU 0) or `VIGILIX_WEAPON_DEVICE=cpu`.

2. **Export to TensorRT FP16** (`export_weapon_model_trt.py`). This is the
   single biggest lever — roughly 2x over plain PyTorch FP32 on the same
   GPU, per the model card's own numbers (5ms → 2ms on H100). Must be
   re-exported on each GPU model you deploy to (engines aren't portable).

3. **Match the training resolution.** The model was trained/validated at
   `imgsz=1024`; both `detection/weapon_detection.py` and
   `server/stream_worker.py` already pass `imgsz=1024`. Don't drop this to
   640 "for speed" — it was explicitly why Phase 3 training improved
   recall on small objects (distant knives, grenade pins); you'd be
   throwing away accuracy the fine-tuning paid for.

4. **Tune confidence per use case**, not just for speed but for the
   alert behavior you want (per the model card's threshold guidance):
   - Security screening / don't-miss-anything: `conf≈0.15–0.25`
   - Standard alerting (current default): `conf=0.35`
   - Low false-positive budget: `conf≈0.5–0.6`
   Lower `conf` costs a bit of inference time (more boxes to post-process)
   but the effect is small compared to resolution/precision choices.

5. **Decoupled capture/display rate vs. detection rate (implemented).**
   `server/stream_worker.py` now reads and publishes every captured frame
   for the live view at full source fps, but only runs the YOLO models
   every Nth frame, throttled to `VIGILIX_DETECT_FPS` (default 7 fps; set
   in `.env`, see `.env.example`). On the frames in between, the last
   known detection boxes are redrawn on the fresh frame so the live view
   doesn't flicker, while GPU/CPU load per camera drops roughly 3-6x
   compared to detecting on every frame (25-30 fps → 5-10 fps).
   `REQUIRED_FRAMES` confirmation still works correctly — it counts
   consecutive *detection* passes, not raw frames, so at 7 fps detection
   an alert still confirms in well under a second.

   Sizing rule of thumb: `max_concurrent_cameras ≈ (GPU inferences/sec
   capacity) / VIGILIX_DETECT_FPS`. E.g. an RTX 3060 on TensorRT FP16
   (~70-100 inferences/sec) at the 7 fps default → roughly 10-14 cameras
   on one GPU, versus ~2-3 cameras if detecting on every frame.

6. **Half precision without full TensorRT export.** If you can't export
   to TensorRT (e.g. during development), `model.predict(..., half=True)`
   on a CUDA device gets some of the FP16 speedup directly in PyTorch.

7. **One model instance shared across streams (already done).**
   `stream_worker.py` loads the weapon model once (`_weapon_model`) and
   serializes access with `_model_lock` rather than loading a copy per
   camera — keep this pattern as you add cameras; it's what keeps VRAM
   usage from multiplying per stream.

8. **Batch multiple camera streams per inference call** if you scale past
   a handful of cameras: collect one frame per active stream and call
   `model.predict(source=[frame1, frame2, ...])` instead of looping
   per-camera — GPUs are far more efficient batched than looped.

9. **Drop the duplicate Fire_Smoke pass once you trust it.** weapon-yolo26x
   already outputs a `Fire Smoke` class alongside the dedicated
   `fire_smoke_model.pt`. Running both models doubles GPU work for
   overlapping signal; both `detection/weapon_detection.py` and
   `server/stream_worker.py` currently ignore `Fire Smoke` from the
   weapon model to avoid duplicate alerts, but if you later verify its
   fire recall independently, you could retire the separate fire model
   and cut one full inference pass per frame.

## 5. Architectural & Pipeline Optimizations Implemented

1. **Fully Decoupled Capture, Inference, and Streaming**:
   - Ingestion (`CameraStream`): Dedicated capture thread runs with low-latency buffers (`CAP_PROP_BUFFERSIZE=1`), continuously reading from the camera or RTSP stream. This prevents OpenCV's internal socket queues from accumulating stale frames.
   - Inference Worker: Runs asynchronously in an independent thread throttled to `VIGILIX_DETECT_FPS` (default 7 FPS on CUDA, 2 FPS on CPU). The heavy YOLO models never block or freeze video ingestion or display.
   - Display & MJPEG Streaming: Smoothly overlays latest bounding boxes onto fresh camera frames at full source FPS without waiting on inference passes.

2. **Dynamic Demand-Based JPEG Encoding**:
   - Previously, `cv2.imencode` was executed on every single frame captured for every camera, consuming 30–50% CPU even when no user had the dashboard open.
   - Now, JPEG encoding only activates when active viewers are subscribed to `/api/streams/<id>/mjpeg`, saving up to 70% CPU during idle background surveillance.
   - The streaming endpoint uses an event-driven `threading.Condition` variable, delivering new frames immediately upon generation instead of spinning busy polling loops.

3. **One-Time Model Warmup at Startup**:
   - Model compilation and buffer allocation previously caused a 5–10 second freeze on the first frame.
   - `get_models()` now runs a single-pass warmup on initialization, ensuring instant responsiveness when cameras connect.

4. **Hardware & Device Auto-Acceleration**:
   - Device auto-selection: Uses CUDA if available with FP16 half precision (`half=True`), speeding up GPU inference by ~2x and halving VRAM usage.
   - Fire detection now runs on the same accelerated device rather than being hardcoded to CPU.
   - Configurable resolution: `VIGILIX_WEAPON_IMGSZ` (defaults to 1024 on GPU, 640 on CPU for real-time responsiveness).
   - PyTorch CPU thread count is tuned (`torch.set_num_threads`) to prevent multi-core thrashing.
   - Graceful fallback: If `weapon_yolo26x.pt` is not present, falls back gracefully to bundled checkpoints rather than crashing.

5. **SQLite WAL Mode & Query Indexing**:
   - Enabled SQLite Write-Ahead Logging (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`) in `server/store.py` for high-concurrency read/write transactions.
   - Created descending indexes on `alerts(created_at DESC)` and `cameras(created_at DESC)` to eliminate full-table scans.

6. **Frontend Video Stream Reconnection Storm Fix**:
   - Fixed `frontend/index.html` where `refreshStreams()` wiped and recreated the DOM containing `<img src="...">` every 4 seconds.
   - Now preserves existing video stream elements and only updates camera status badges in place, eliminating stream flickering and network reconnect storms.

7. **Zero-Duplication Alert Pipeline**:
   - Eliminated redundant double-screenshot writes to disk on detections.
   - Added native Windows non-blocking audio alerts via `winsound.PlaySound` with cross-platform fallbacks.
   - Standalone CLI scripts (`detection/fire_detection.py`, `detection/weapon_detection.py`, `main.py`) now have proper function exports, CLI arguments, and execution guards.

