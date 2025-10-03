from openvino import Core
import numpy as np
from PIL import Image
import yaml
from pathlib import Path


from .utils import preprocess



class OVClfModel:
    def __init__(self, model_path: str, topk:int=1, device: str = "CPU"):
        self.core = Core()
        self.topk = topk
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # Load metadata
        meta_path = Path(model_path).with_name("metadata.yaml")
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
        self.task = meta.get("task", "detect")
        self.imgsz = tuple(meta.get("imgsz", [640, 640]))
        self.class_names = meta.get("names", {})


    def postprocess(self, output:np.ndarray, class_names:dict[int, str], topk:int) -> list[dict]:
        probs = output[0]  # (num_classes,)
        topk = min(topk, probs.shape[0])
        top_indices = np.argsort(-probs)[:topk]
        results = []
        for idx in top_indices:
            cls_id = int(idx)
            results.append({
                "cls_index": cls_id,
                "cls_name": class_names.get(cls_id, str(cls_id)),
                "confidence": round(float(probs[idx]), 3)
            })
        return results


    def predict(self, pil_image:Image.Image, topk:int=None) -> list[dict]:
        """Perform item detection on the input image"""
        assert isinstance(pil_image, Image.Image)
        topk = topk or self.topk
        inp, _, _ = preprocess(pil_image, self.imgsz)
        raw = self.compiled_model(inp)[self.output_layer]
        return self.postprocess(raw, self.class_names, topk=topk)


    def __call__(self, pil_image: Image.Image, topk:float=None)-> list[dict]:
        """Perform object detection on the input image"""
        return self.predict(pil_image=pil_image, topk=topk)
    



if __name__ == "__main__":
    import openvino as ov
    from ultralytics import YOLO
    import os

    #base model folder path
    base_model_path ="yolo11s-cls.pt"
    base_model_id = os.path.splitext(base_model_path)[0]

    #load reference model
    model = YOLO(base_model_path)

    # Compress model and create a folder {base_model_id}_openvino_model/ containinng
    # - {base_model_id}.xml
    # - {base_model_id}.bin
    # - metadata.yaml
    model.export(format="openvino", dynamic=False, simplify=True, opset=12)

    # Export model from dev to production folder
    source_path = f"{base_model_id}_openvino_model/"
    export_path = source_path.replace("/dev/", "/prod/")
    os.makedirs(export_path,exist_ok=True)
    os.system(f"mv {source_path} {export_path}")