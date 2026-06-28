"""
train.py - Fine-tune YOLOv8n on a custom dataset.

This script is designed to run on Google Colab (no local GPU required) but
works equally well on any machine with a CUDA-capable GPU or even on CPU for
small datasets.

─── Quick-start on Google Colab ────────────────────────────────────────────
# 1. Upload this file and your dataset, then:
!pip install ultralytics roboflow

# 2. (Option A) Download from Roboflow:
!python train.py --data /path/to/data.yaml --roboflow-key <KEY> \
                 --workspace <WS> --project <PROJ> --version 1

# 3. (Option B) Use a local dataset you already have:
!python train.py --data ./dataset/data.yaml

# 4. Find the fine-tuned weights at:
#    runs/detect/surveillance_finetune/weights/best.pt
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Roboflow optional helper (only needed when --roboflow-key is provided)     #
# --------------------------------------------------------------------------- #
def maybe_download_roboflow(args: argparse.Namespace) -> str:
    """Download a Roboflow dataset and return the path to data.yaml."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] Install roboflow: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=args.roboflow_key)
    dataset = (
        rf.workspace(args.workspace)
          .project(args.project)
          .version(args.version)
          .download("yolov8", location="./dataset")
    )
    yaml_path = str(Path(dataset.location) / "data.yaml")
    print(f"[INFO] Dataset ready: {yaml_path}")
    return yaml_path


# --------------------------------------------------------------------------- #
#  Training                                                                    #
# --------------------------------------------------------------------------- #
def run_training(data_yaml: str, args: argparse.Namespace) -> Path:
    """Fine-tune YOLOv8n on the given dataset.

    Key hyperparameters:
      epochs      – Number of complete passes over the training set.
      imgsz       – Input image size (square).  640 is the standard YOLO size.
      batch       – Batch size.  -1 = auto (uses ~60 % of available VRAM).
      lr0         – Initial learning rate.
      lrf         – Final learning rate as a fraction of lr0.
      patience    – Early-stopping patience (epochs without improvement).
      freeze      – Freeze the first N backbone layers to speed up training
                    and reduce over-fitting on small datasets.
      augment     – Enable Mosaic, MixUp, and other augmentations.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Install ultralytics: pip install ultralytics")
        sys.exit(1)

    # Load the pre-trained YOLOv8 nano weights as the starting point
    pretrained = args.pretrained
    print(f"[INFO] Loading base weights: {pretrained}")
    model = YOLO(pretrained)

    print(f"[INFO] Starting fine-tuning for {args.epochs} epochs…")
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        freeze=args.freeze,
        augment=True,
        mosaic=1.0,          # Mosaic augmentation probability
        mixup=0.1,           # MixUp augmentation probability
        degrees=10.0,        # Random rotation ±10°
        translate=0.1,       # Random translate ±10 %
        scale=0.5,           # Random scale ±50 %
        flipud=0.0,          # No vertical flip (not useful for surveillance)
        fliplr=0.5,          # 50 % horizontal flip
        name="surveillance_finetune",
        project="runs/detect",
        exist_ok=True,
        device=args.device,
        workers=args.workers,
        amp=True,            # Automatic mixed precision (halves VRAM usage)
        verbose=True,
    )

    best_weights = Path("runs/detect/surveillance_finetune/weights/best.pt")
    return best_weights


# --------------------------------------------------------------------------- #
#  Post-training: copy best weights to models/ folder                         #
# --------------------------------------------------------------------------- #
def copy_to_models(weights_path: Path, models_dir: Path) -> None:
    """Copy the best fine-tuned weights to the project models/ directory."""
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / "yolov8n_finetuned.pt"
    shutil.copy2(weights_path, dest)
    print(f"[INFO] Best weights copied to: {dest}")


# --------------------------------------------------------------------------- #
#  Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def run_evaluation(weights_path: Path, data_yaml: str, args: argparse.Namespace) -> None:
    """Validate the fine-tuned model and print mAP metrics."""
    from ultralytics import YOLO

    print("\n[INFO] Evaluating fine-tuned model on validation set…")
    model = YOLO(str(weights_path))
    metrics = model.val(data=data_yaml, imgsz=args.imgsz, device=args.device, verbose=True)
    print(f"\n  mAP50   : {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall   : {metrics.box.mr:.4f}")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n for the CSCI435 Surveillance System."
    )
    # Dataset source
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data", metavar="YAML", default=None,
                       help="Path to data.yaml (use this for a local dataset)")
    group.add_argument("--roboflow-key", metavar="KEY",
                       help="Roboflow API key (downloads dataset automatically)")

    # Roboflow details
    parser.add_argument("--workspace", metavar="WS", default="")
    parser.add_argument("--project", metavar="PROJ", default="")
    parser.add_argument("--version", metavar="VER", type=int, default=1)

    # Training hyperparameters
    parser.add_argument("--pretrained", default="yolov8n.pt",
                        help="Base weights to start fine-tuning from")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size in pixels (default 640)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size; -1 for auto (default -1)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate (default 0.01)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final LR fraction of lr0 (default 0.01)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience in epochs (default 20)")
    parser.add_argument("--freeze", type=int, default=10,
                        help="Freeze first N backbone layers (default 10)")
    parser.add_argument("--device", default="",
                        help="Device: '' (auto), 'cpu', '0', 'cuda:0'")
    parser.add_argument("--workers", type=int, default=4,
                        help="Dataloader worker threads (default 4)")
    parser.add_argument("--models-dir", default="../models",
                        help="Where to copy the best weights after training")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation after training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine data.yaml path
    if args.roboflow_key:
        data_yaml = maybe_download_roboflow(args)
    elif args.data:
        data_yaml = args.data
    else:
        print("[ERROR] Provide --data <yaml> or --roboflow-key <key>.")
        sys.exit(1)

    if not Path(data_yaml).exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        sys.exit(1)

    # Train
    best_weights = run_training(data_yaml, args)

    if not best_weights.exists():
        print("[ERROR] Training did not produce a best.pt – check the run logs.")
        sys.exit(1)

    # Copy to models/
    models_dir = Path(args.models_dir)
    copy_to_models(best_weights, models_dir)

    # Evaluate
    if not args.no_eval:
        run_evaluation(best_weights, data_yaml, args)

    print("\n[DONE] Fine-tuning complete.")
    print(f"       Use the model: {(models_dir / 'yolov8n_finetuned.pt').resolve()}")
    print("       Update detection.py / tracking.py to point at the new weights.")


if __name__ == "__main__":
    main()
