"""
dataset_prep.py - Dataset Preparation for YOLOv8 Fine-Tuning.

This script helps you download and organise a dataset from Roboflow (or any
existing YOLO-format dataset) into the directory layout that Ultralytics
expects:

    dataset/
      images/
        train/   ← training images
        val/     ← validation images
      labels/
        train/   ← YOLO .txt annotation files (one per image)
        val/
      data.yaml  ← dataset configuration consumed by train.py

Usage (Roboflow dataset):
    python dataset_prep.py --roboflow-key <API_KEY> \
                           --workspace <WORKSPACE> \
                           --project <PROJECT_NAME> \
                           --version <VERSION_NUMBER> \
                           --output ./dataset

Usage (existing local YOLO dataset):
    python dataset_prep.py --local-dataset /path/to/existing/dataset \
                           --output ./dataset
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Roboflow download                                                           #
# --------------------------------------------------------------------------- #
def download_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: Path,
) -> Path:
    """Download a Roboflow dataset in YOLOv8 format.

    Requires: pip install roboflow
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow package not found.  Run: pip install roboflow")
        sys.exit(1)

    print(f"[INFO] Connecting to Roboflow workspace '{workspace}'…")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location=str(output_dir))
    print(f"[INFO] Dataset downloaded to: {dataset.location}")
    return Path(dataset.location)


# --------------------------------------------------------------------------- #
#  Local dataset copy / validation                                             #
# --------------------------------------------------------------------------- #
def prepare_local_dataset(src: Path, dst: Path) -> None:
    """Copy and validate an existing YOLO-format dataset to *dst*.

    Expected *src* structure:
        src/
          images/train/  images/val/
          labels/train/  labels/val/
          data.yaml
    """
    required = [
        src / "images" / "train",
        src / "images" / "val",
        src / "labels" / "train",
        src / "labels" / "val",
    ]
    for p in required:
        if not p.exists():
            print(f"[ERROR] Missing required directory: {p}")
            sys.exit(1)

    print(f"[INFO] Copying dataset from {src} → {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print("[INFO] Copy complete.")


# --------------------------------------------------------------------------- #
#  Validation helpers                                                          #
# --------------------------------------------------------------------------- #
def validate_dataset(dataset_dir: Path) -> None:
    """Check image/label counts and print a summary."""
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            print(f"[WARN] {img_dir} does not exist.")
            continue

        images = list(img_dir.glob("*.[jJ][pP][gG]")) + \
                 list(img_dir.glob("*.[pP][nN][gG]")) + \
                 list(img_dir.glob("*.[bB][mM][pP]"))

        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        print(f"  [{split}]  {len(images)} images  |  {len(labels)} label files")

        # Check for images without a matching label file
        missing = [
            img.stem for img in images
            if not (lbl_dir / (img.stem + ".txt")).exists()
        ]
        if missing:
            print(f"    [WARN] {len(missing)} images have no label file (first 5: {missing[:5]})")


def generate_data_yaml(dataset_dir: Path, class_names: list[str]) -> Path:
    """Write a data.yaml file if one is not already present."""
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.exists():
        print(f"[INFO] data.yaml already exists at {yaml_path}")
        return yaml_path

    nc = len(class_names)
    names_str = "\n".join(f"  - {n}" for n in class_names)
    content = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {nc}\n"
        f"names:\n{names_str}\n"
    )
    yaml_path.write_text(content)
    print(f"[INFO] data.yaml written to {yaml_path}")
    return yaml_path


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLO dataset for fine-tuning.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--roboflow-key", metavar="KEY", help="Roboflow API key")
    group.add_argument("--local-dataset", metavar="PATH", help="Path to an existing YOLO dataset")

    # Roboflow-specific arguments
    parser.add_argument("--workspace", metavar="WS", default="", help="Roboflow workspace slug")
    parser.add_argument("--project", metavar="PROJ", default="", help="Roboflow project slug")
    parser.add_argument("--version", metavar="VER", type=int, default=1, help="Dataset version number")

    parser.add_argument("--output", metavar="PATH", default="./dataset", help="Destination directory")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["object"],
        help="Class names (only needed if data.yaml is missing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)

    if args.roboflow_key:
        dataset_dir = download_roboflow(
            api_key=args.roboflow_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            output_dir=output_dir,
        )
    else:
        prepare_local_dataset(Path(args.local_dataset), output_dir)
        dataset_dir = output_dir

    print("\n[INFO] Dataset summary:")
    validate_dataset(dataset_dir)

    generate_data_yaml(dataset_dir, args.classes)

    print(f"\n[DONE] Dataset ready at: {dataset_dir.resolve()}")
    print("       Run train.py with:  --data", dataset_dir / "data.yaml")


if __name__ == "__main__":
    main()
