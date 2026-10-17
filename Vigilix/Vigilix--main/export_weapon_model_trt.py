"""
Exports model/detection/weapon_yolo26x.pt to a TensorRT FP16 engine for
much faster GPU inference (see OPTIMIZATION.md).

Requirements: NVIDIA GPU, CUDA, and `pip install tensorrt onnx onnxsim`.
The .engine file is NOT portable across GPU architectures - re-run this
script on the machine that will actually run inference (the model card's
own .engine was built on an H100 and won't load on, e.g., a T4 or RTX card).

Usage:
    python export_weapon_model_trt.py
"""

from ultralytics import YOLO

SRC_WEIGHTS = "model/detection/weapon_yolo26x.pt"
IMAGE_SIZE = 1024  # must match the resolution the model was trained at

model = YOLO(SRC_WEIGHTS)

engine_path = model.export(
    format="engine",
    imgsz=IMAGE_SIZE,
    half=True,      # FP16 - ~2x faster than FP32 with negligible accuracy loss
    device=0,        # GPU index to build the engine on/for
    workspace=6,      # GB of GPU memory the builder may use
)

print("TensorRT engine written to:", engine_path)
print(
    "Point WEAPON_MODEL_PATH (server/stream_worker.py) or MODEL_PATH "
    "(detection/weapon_detection.py) at this .engine file to use it. "
    "Rebuild this engine on every machine/GPU model you deploy to."
)
