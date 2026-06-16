#!/usr/bin/env python3
"""
Train a YOLOv8n detector for penis + face, export to ONNX.
Run once; produces detector.onnx — drop it in the FunscriptFlow repo root.

Prerequisites:
    pip install ultralytics onnx

Usage:
    # Option A: train from scratch with your own labeled dataset
    python train_detector.py --data /path/to/dataset/data.yaml --epochs 50

    # Option B: use a pre-labeled dataset (if available)
    python train_detector.py --dataset nsfw-default --epochs 50

    # Option C: download a pre-trained model from HuggingFace
    python train_detector.py --download

Output: detector.onnx in current directory

Dataset format (YOLO):
    dataset/
      data.yaml          # class names + paths
      train/images/      # .jpg training images
      train/labels/      # .txt labels (YOLO format: class x_center y_center w h)
      val/images/
      val/labels/
"""

import argparse, os, sys, shutil


def train(yaml_path, epochs, imgsz):
    from ultralytics import YOLO

    print(f"Training YOLOv8n on {yaml_path} for {epochs} epochs...")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        device="cpu",  # change to 0 for GPU
        workers=2,
        verbose=True,
    )

    print("Exporting to ONNX...")
    model.export(format="onnx", imgsz=imgsz, opset=12)
    src = "runs/detect/train/weights/best.onnx"
    if os.path.exists(src):
        shutil.copy(src, "detector.onnx")
        print("Done: detector.onnx ready")
    else:
        print(f"WARNING: ONNX model not found at {src}, check training output.")


def download_pretrained():
    """
    Attempt to download a pre-trained model from HuggingFace.
    Update the repo/URL as better models become available.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("pip install huggingface_hub first")
        sys.exit(1)

    # Placeholder — replace with actual repo when available
    repo = "USERNAME/nsfw-detector"
    filename = "yolov8n-detector.onnx"

    print(f"Downloading {filename} from {repo}...")
    path = hf_hub_download(repo_id=repo, filename=filename)
    shutil.copy(path, "detector.onnx")
    print(f"Done: {os.path.abspath('detector.onnx')}")


def create_template_dataset():
    """Create a template data.yaml for the user to fill in."""
    template = """# YOLO dataset config
# Classes: 0 = penis, 1 = face
path: ./
train: train/images
val: val/images

names:
  0: penis
  1: face
"""
    os.makedirs("dataset/train/images", exist_ok=True)
    os.makedirs("dataset/train/labels", exist_ok=True)
    os.makedirs("dataset/val/images", exist_ok=True)
    os.makedirs("dataset/val/labels", exist_ok=True)
    with open("dataset/data.yaml", "w") as f:
        f.write(template)
    print("Template dataset created at ./dataset/")
    print("Place your images in train/images/ and labels in train/labels/")
    print("Labels must be YOLO format: class x_center y_center width height (normalized 0-1)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n penis+face detector")
    parser.add_argument("--data", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--download", action="store_true", help="Download pre-trained model")
    parser.add_argument("--template", action="store_true", help="Create template dataset structure")
    args = parser.parse_args()

    if args.template:
        create_template_dataset()
        sys.exit(0)

    if args.download:
        download_pretrained()
        sys.exit(0)

    if args.data:
        train(args.data, args.epochs, args.imgsz)
        sys.exit(0)

    parser.print_help()
    print("\nFirst time? Run: python train_detector.py --template")
