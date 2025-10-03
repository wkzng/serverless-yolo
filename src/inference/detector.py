from openvino import Core
import numpy as np
from PIL import Image
import yaml
from pathlib import Path

from .utils import preprocess, scale_boxes



class OVDetModel:
    def __init__(self, model_path: str, conf_thres:float=0.25, device: str="CPU"):
        """
        Initializes the OpenVINO detection model for lightweight inference.
        This class is dependency-free from PyTorch and Ultralytics.

        Args:
            model_path (str): Path to the .xml file of the OpenVINO model.
            conf_thres (float): Confidence threshold for filtering detections.
            device (str): The device to run inference on (e.g., "CPU", "GPU").
        """
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.conf_thres = conf_thres

        # Load metadata.yaml next to the XML model
        meta_path = Path(model_path).with_name("metadata.yaml")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)
            self.imgsz = tuple(meta.get("imgsz", [640, 640]))
            self.class_names = meta.get("names", {})
            self.stride = meta.get("stride", 32)
        else:
            # Fallbacks
            self.imgsz = (640, 640)
            self.class_names = {}
            self.stride = 32


    def postprocess(self, output:np.ndarray, input_hw:tuple[int], orig_img, conf_thres:float, ratio_pad=None):
        # output shape: (max_det, 6) = [x1,y1,x2,y2,score,class]
        dets = output[output[:, 4] > conf_thres]
        if not len(dets):
            return np.zeros((0, 6))
        # rescale from input shape -> original shape
        dets[:, :4] = scale_boxes(input_hw, dets[:, :4], orig_img.shape, ratio_pad=ratio_pad)
        return dets


    def predict(self, pil_image: Image.Image, conf_thres:float=None) -> list[dict]:
        """Perform item detection on the input image"""
        assert isinstance(pil_image, Image.Image)
        conf_thres = conf_thres or self.conf_thres

        inp, orig_img, (ratio, (dw, dh)) = preprocess(pil_image, self.imgsz)
        raw = self.compiled_model(inp)[self.output_layer][0]
        detections = self.postprocess(raw, inp.shape[2:], orig_img, ratio_pad=(ratio, (dw, dh)), conf_thres=conf_thres)

        # Map class IDs to names
        results = []
        for x1, y1, x2, y2, score, cls_id in detections:
            label = self.class_names.get(int(cls_id), str(int(cls_id)))
            results.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(score, 3),
                "cls_index": int(cls_id),
                "cls_name": label,
            })
        return results
    

    def __call__(self, pil_image: Image.Image, conf_thres:float=None)-> list[dict]:
        """Perform object detection on the input image"""
        return self.predict(pil_image=pil_image, conf_thres=conf_thres)
    