import argparse
import os
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing 'ultralytics'
# This is necessary because we are running this script from a subdirectory.
sys.path.append(str(Path(__file__).resolve().parents[1]))


try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' package not found.")
    print("Please install the development dependencies: pip install -r requirements-dev.txt")
    sys.exit(1)


def export_model(model_path: str, task: str):
    """
    Exports a trained YOLO model to the OpenVINO format.

    This script handles the "heavy" part of the process, requiring
    the full 'ultralytics' and 'torch' libraries. The output is a
    lightweight, dependency-free OpenVINO model ready for deployment.

    Args:
        model_path (str): Path to the input .pt model file.
        task (str): The type of model task. Must be one of 'detect', 'classify', 'pose'.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    # Common export arguments
    export_args = {
        "format": "openvino",
        "dynamic": False,
        "simplify": True,
        "opset": 12
    }

    # Task-specific arguments. Detection and Pose models benefit from
    # including Non-Maximum Suppression (NMS) in the exported graph.
    if task in ["detect", "pose"]:
        export_args["nms"] = True

    print(f"Exporting model for task '{task}' with arguments: {export_args}")
    
    # The export() method creates a new directory named e.g., 'yolov11n-face_openvino_model'
    output_dir_name = model.export(**export_args)
    
    print(f"\nExport successful!")
    print(f"OpenVINO model, weights, and metadata have been saved to: '{output_dir_name}'")
    print("\nThis folder contains everything needed for production deployment.")
    print("You can now copy this folder to your production environment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO models to OpenVINO format.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained YOLO .pt model file."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["detect", "classify", "pose"],
        help="The task type of the model."
    )

    args = parser.parse_args()
    export_model(args.model_path, args.task)

    # Example Usage:
    # python scripts/export_model.py --model-path models/yolo11n.pt --task detect
    # python scripts/export_model.py --model-path models/yolo11n-cls.pt --task classify
    # python scripts/export_model.py --model-path models/yolo11n-pose.pt --task pose
