import numpy as np
import cv2
from typing import Tuple
from PIL import Image


# -----------------------------
# 🔹 Image Preprocessing
# -----------------------------

def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = False,
    stride: int = 32,
):
    """
    Resize image and padding for detection. Takes image as input,
    resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints

    Parameters:
      img (np.ndarray): image for preprocessing
      new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
      color (Tuple(int, int, int)): color for filling padded area
      auto (bool): use dynamic input size, only padding for stride constrins applied
      scale_fill (bool): scale image to fill new_shape
      scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
      stride (int): input padding stride
    Returns:
      img (np.ndarray): image after preprocessing
      ratio (Tuple(float, float)): hight and width scaling ratio
      padding_size (Tuple(int, int)): height and width padding size
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)




def preprocess(pil_img: Image.Image, new_shape=(640, 640)):
    """
    Convert PIL → np, apply letterbox, normalize, transpose.
    Returns (input_tensor, orig_img, (gain, pad)).
    """
    img = np.array(pil_img)
    img_resized, ratio, (dw, dh) = letterbox(img, new_shape)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = img_resized.transpose(2, 0, 1)  # HWC → CHW
    input_tensor = np.expand_dims(img_resized, 0)  # add batch
    return input_tensor, img, (ratio, (dw, dh))


# -----------------------------
# 🔹 Box / Keypoint Scaling
# -----------------------------

def clip_boxes(boxes, shape):
    """
    Clip bounding boxes to image boundaries.
    Args:
        boxes (torch.Tensor | np.ndarray): Bounding boxes to clip.
        shape (tuple): Image shape as (height, width).

    Returns:
        (np.ndarray): Clipped bounding boxes.
    """
    if not isinstance(boxes, np.ndarray):
        raise ValueError("boxes should be np.array objects")
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes.round()


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding: bool = True, xywh: bool = False):
    """
    Rescale bounding boxes from one image shape to another.

    Rescales bounding boxes from img1_shape to img0_shape, accounting for padding and aspect ratio changes.
    Supports both xyxy and xywh box formats.

    Args:
        img1_shape (tuple): Shape of the source image (height, width).
        boxes (torch.Tensor): Bounding boxes to rescale in format (N, 4).
        img0_shape (tuple): Shape of the target image (height, width).
        ratio_pad (tuple, optional): Tuple of (ratio, pad) for scaling. If None, calculated from image shapes.
        padding (bool): Whether boxes are based on YOLO-style augmented images with padding.
        xywh (bool): Whether box format is xywh (True) or xyxy (False).

    Returns:
        (torch.Tensor): Rescaled bounding boxes in the same format as input.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)



def scale_keypoints(keypoints: np.ndarray, input_hw:tuple[int], orig_shape:tuple[int], ratio_pad:tuple[float]=None) -> np.ndarray:
    """
    Rescale keypoints from input size back to original image size.
    keypoints: (num_kpts, 3) with (x,y,conf)
    input_hw: (h, w) of model input
    orig_shape: (h, w, c) of original image
    ratio_pad: (ratio, pad) from preprocess
    """
    if ratio_pad is None:
        gain = min(input_hw[0] / orig_shape[0], input_hw[1] / orig_shape[1])
        pad = (
            (input_hw[1] - orig_shape[1] * gain) / 2,
            (input_hw[0] - orig_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # undo padding and scaling
    kpts = keypoints.copy()
    kpts[:, 0] -= pad[0]  # x
    kpts[:, 1] -= pad[1]  # y
    kpts[:, :2] /= gain

    # clip to image size
    kpts[:, 0] = np.clip(kpts[:, 0], 0, orig_shape[1])
    kpts[:, 1] = np.clip(kpts[:, 1], 0, orig_shape[0])
    return kpts