from openvino import Core
import numpy as np
from PIL import Image


from .utils import preprocess, scale_boxes, scale_keypoints


def postprocess_pose_ov(output, input_hw, orig_img, class_names, ratio_pad=None, conf_thres=0.25):
    """
    Postprocess YOLO pose output (OpenVINO-exported).
    Output shape: (N, 57) -> [x1,y1,x2,y2,score,class_id, keypoints...]
    """
    # unwrap to (N,57)
    if isinstance(output, (list, tuple)):
        dets = output[0]
    else:
        dets = output

    dets = np.array(dets)
    if dets.ndim == 3 and dets.shape[0] == 1:
        dets = dets[0]

    if dets.ndim != 2 or dets.shape[1] < 57:
        raise ValueError(f"Unexpected pose output shape: {dets.shape}")

    # filter by confidence
    dets = dets[dets[:, 4] > conf_thres]
    results = []

    for det in dets:
        x1, y1, x2, y2, conf, cls_id = det[:6]
        keypoints = det[6:].reshape(-1, 3)

        # rescale bbox
        bbox = np.array([[x1, y1, x2, y2]])
        bbox = scale_boxes(input_hw, bbox, orig_img.shape, ratio_pad=ratio_pad)[0]
        bbox = [int(value) for value in bbox]

        # rescale keypoints
        keypoints = scale_keypoints(keypoints, input_hw, orig_img.shape, ratio_pad=ratio_pad)

        results.append({
            "cls_index": int(cls_id),
            "cls_name": class_names.get(int(cls_id), str(int(cls_id))),
            "confidence": round(float(conf), 3),
            "box": bbox,
            "keypoints": keypoints.tolist()
        })

    return results


class OVPoseModel:
    def __init__(self, model_path: str, device: str = "CPU", conf_thres:float=0.25, class_names: dict[int, str] | None = None):
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.class_names = class_names or {}
        self.conf_thres = conf_thres

    def predict(self, pil_image: Image.Image, conf_thres:float=None) -> list[dict]:
        """Perform pose estimation on people present on the image"""
        assert isinstance(pil_image, Image.Image)
        conf_thres = conf_thres or self.conf_thres
    
        inp, orig_img, ratio_pad = preprocess(pil_image, new_shape=(640, 640))
        raw = self.compiled_model(inp)[self.output_layer]
        results = postprocess_pose_ov(raw, inp.shape[2:], orig_img, self.class_names, ratio_pad, conf_thres)
        return results

    def __call__(self, pil_image: Image.Image, conf_thres:float=None)-> list[dict]:
        """Perform pose estimation on people present on the image"""
        return self.predict(pil_image=pil_image, conf_thres=conf_thres)
    


if __name__ == "__main__":
    import openvino as ov
    from ultralytics import YOLO
    import os

    #base model folder path
    base_model_path ="yolo11-pose.pt"
    base_model_id = os.path.splittext(base_model_path)[0]

    #load reference model
    model = YOLO(base_model_path)

    # Compress model and create a folder {base_model_id}_openvino_model/ containinng
    # - {base_model_id}.xml
    # - {base_model_id}.bin
    # - metadata.yaml
    model.export(format="openvino", dynamic=False, simplify=True, opset=12, nms=True)

    # yolo11n_openvino_model/
    # ├── yolo11n.xml
    # ├── yolo11n.bin
    # └── metadata.yaml