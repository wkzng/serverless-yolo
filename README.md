# Serverless YOLO: A Framework for Production-Ready, PyTorch-Free Inference

This repository provides a complete, end-to-end framework for converting any **Ultralytics YOLOv8 model** (Detection, Classification, Pose Estimation) into the lightweight **OpenVINO** format and deploying it as a scalable, low-cost serverless function on **AWS Lambda**.

---

## The Problem This Solves

Deploying PyTorch models directly into serverless environments like AWS Lambda is often impossible due to:

* **Massive Dependencies**: PyTorch and related libraries (ultralytics, torchvision, etc.) are large and easily exceed serverless package size limits.
* **Cold Start Latency**: Loading these heavy frameworks causes slow response times.
* **Cost**: Larger memory and package sizes result in higher operational costs.

This project demonstrates a core **MLOps principle**: separating the heavy training/conversion environment from the lightweight, optimized production environment.

---

## Project Structure

```
├── models/                 # Holds exported OpenVINO models
├── notebooks/
│   └── test_inference.ipynb  # Notebook for testing production-ready code
├── scripts/
│   └── export_model.py     # (Dev) Convert .pt models to OpenVINO
├── src/
│   └── inference/          # (Prod) Lightweight, PyTorch-free inference code
│       ├── classifier.py
│       ├── detector.py
│       ├── pose_estimator.py
│       └── utils.py
├── Dockerfile              # Defines serverless container environment
├── lambda_function.py      # Handler code for AWS Lambda
├── requirements-dev.txt    # Heavy dependencies (conversion)
└── requirements-prod.txt   # Lightweight dependencies (inference)
```

---

## The Two-Stage Workflow

### **Stage 1: Model Preparation (Development Environment)**

This stage requires heavy libraries (`ultralytics`, `torch`, `openvino-dev`) to convert your trained `.pt` models.

**Setup the Development Environment:**

```bash
pip install -r requirements-dev.txt
```

**Run the Export Script:**

```bash
python scripts/export_model.py \
    --model-path "path/to/your/yolov8n.pt" \
    --task "detect" \
    --output-dir "models"
```

This will create a `yolov8n_openvino_model` folder inside `models/` containing:

* `.xml`
* `.bin`
* `metadata.yaml`

---

### **Stage 2: Inference & Deployment (Production Environment)**

This stage uses a minimal, lightweight environment. It only needs OpenVINO and basic dependencies.

**Setup the Production Environment:**

```bash
pip install -r requirements-prod.txt
```

**Local Testing:**
Before deploying, test your exported models locally:

```bash
# Start Jupyter from project root
jupyter notebook
```

Open and run:

```
notebooks/test_inference.ipynb
```

---

## Deploying to AWS Lambda

The included **Dockerfile** and **src/lambda_function.py** allow deployment as a serverless function.

### **Build the Docker Image**

```bash
docker build -t serverless-yolo-inference .
```

### **Push to Amazon ECR**

1. Create a repository in ECR.
2. Authenticate Docker to your registry.
3. Tag and push your image (follow AWS console commands).

### **Create & Configure the Lambda Function**

* In AWS Lambda console:

  * Create new function → "Container image".
  * Select your pushed image.
* Increase defaults:

  * **Memory** → `1024 MB`
  * **Timeout** → `30s`
* Add environment variables:

**Examples:**

Detection:

```
YOLO_TASK=detect
MODEL_XML_PATH=/app/models/yolo11n_openvino_model/yolo11n.xml
```

Classification:

```
YOLO_TASK=classify
MODEL_XML_PATH=/app/models/yolo11n-cls_openvino_model/yolo11n-cls.xml
```

Pose Estimation:

```
YOLO_TASK=pose
MODEL_XML_PATH=/app/models/yolo11n-pose_openvino_model/yolo11n-pose.xml
```

---

## Invoke the Function

Test with API Gateway or Lambda console's "Test" tab.
Send JSON with base64-encoded image:

```json
{
  "image": "<your_base64_encoded_image_string>"
}
```
