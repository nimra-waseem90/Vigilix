"""
Downloads the weapon-detection model weights from Hugging Face:
https://huggingface.co/HaiderKhan6410/weapon-yolo26x

Replaces the old model/detection/gun_knife_yolo11n.pt (2-class, YOLO11n)
with model/detection/weapon_yolo26x.pt (7-class, YOLO26x):
Blunt Weapon, Explosive, Fire Smoke, Firearm, Melee_Weapon, Person, Tool.

Run once before starting the app:
    pip install huggingface_hub
    python download_weapon_model.py
"""

from huggingface_hub import hf_hub_download
import shutil
import os

model_path = hf_hub_download(
    repo_id="HaiderKhan6410/weapon-yolo26x",
    filename="model/best.pt",
)

destination = "model/detection/weapon_yolo26x.pt"

os.makedirs("model/detection", exist_ok=True)

shutil.copy(model_path, destination)

print("Weapon model (YOLO26x) downloaded successfully!")
print("Saved to:", destination)
print(
    "Classes: Blunt Weapon, Explosive, Fire Smoke, Firearm, "
    "Melee_Weapon, Person, Tool"
)
