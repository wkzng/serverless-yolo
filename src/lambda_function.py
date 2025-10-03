import os, json
import io
import base64
from PIL import Image
import traceback
import requests
import numpy as np
from enum import IntEnum
from typing import Any, Callable, Dict


from src.inference.detector import OVDetModel
from src.inference.classifier import OVClfModel
from src.inference.pose_estimator import OVPoseModel



MODELS_CACHE = {}



class HttpStatus(IntEnum):
    OK = 200
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    INTERNAL_ERROR = 500



def make_json_safe(obj):
    """Format the python object to ensure compatiblity with json serilization"""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.complexfloating,)):
        return float(np.abs(obj))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    


def lambda_handler_wrapper(fn: Callable) -> Callable:
    """ Wrap a Lambda handler to catch exceptions and return consistent error responses """
    def wrapped(event: Dict[str, Any], context) -> Dict[str, Any]:
        try:
            return fn(event, context)
        except Exception as e:
            details = traceback.format_exc()
            return {
                "statusCode": HttpStatus.INTERNAL_ERROR,
                "body": json.dumps({
                    "error": "Internal Error",
                    "details": details if os.getenv("STAGE").lower() == 'dev' else None
                })
            }
    return wrapped



def download_image_from_url(url: str) -> Image.Image:
    """Download an image from a given URL"""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))



def base64_to_pil(image_base64: str) -> Image.Image:
    """Convert base64 string to a PIL image"""
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes))



def get_model(model_id:str):
    """Load ro reach preloaded model from cache"""
    model = MODELS_CACHE.get(model_id)
    if model is None:
        if model_id == "yolo11-det":
            model_path = os.getenv("YOLO11_DET_XML_PATH")
            model = OVDetModel(model_path=model_path)
        elif model_id == "yolo11-cls":
            model_path = os.getenv("YOLO11_CLS_XML_PATH")
            model = OVClfModel(model_path=model_path)
        elif model_id == "yolo11-pose":
            model_path = os.getenv("YOLO11_POSE_XML_PATH")
            model = OVPoseModel(model_path=model_path)
        else:
            raise ValueError(f"Unkodn model_id:{model_id}")
        MODELS_CACHE[model_id] = model
        return model
    return model
        


@lambda_handler_wrapper
def lambda_handler(event:dict, context):
    # reach requests
    request_data = json.loads(event["body"]) if "body" in event else event
    #request_data["image_url"] = "https://ultralytics.com/images/bus.jpg"


    # load or download  image
    if request_data.get("image"):
        image = base64_to_pil(request_data["image"])
    elif request_data.get("image_url"):
        image = download_image_from_url(request_data["image_url"])
    else:
        return {"statusCode": HttpStatus.BAD_REQUEST, "body": json.dumps({"error": "IMAGE_REQUIRED"})}

    #process image with detector model
    model = get_model("yolo11-det")
    results = model.predict(image)
    results = make_json_safe(results)

    return {
        "statusCode": HttpStatus.OK, 
        "body": json.dumps({"results": results})
    }