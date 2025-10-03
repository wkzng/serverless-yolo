# Serverless YOLO: A Framework for Production-Ready, PyTorch-Free Inference

This repository provides a complete, end-to-end framework for converting any **Ultralytics YOLO model** (Detection, Classification, Pose Estimation) into the lightweight **OpenVINO** format and deploying it as a scalable, low-cost serverless function on **AWS Lambda**.

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
│   └── lambda_function.py  # Handler code for AWS Lambda
│   └── inference/          # (Prod) Lightweight, PyTorch-free inference code
│       ├── classifier.py
│       ├── detector.py
│       ├── pose_estimator.py
│       └── utils.py
├── Dockerfile              # Defines serverless container environment
├── requirements_dev.txt    # Heavy dependencies (conversion)
└── requirements_prod.txt   # Lightweight dependencies (inference)
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
    --model-path "path/to/your/yolov11n.pt" \
    --task "detect" \
```

This will create a `yolov11n_openvino_model` folder inside `models/` containing:

* `.xml`
* `.bin`
* `metadata.yaml`

Examples:
```bash
python scripts/export_model.py --model-path models/yolo11n.pt --task detect
python scripts/export_model.py --model-path models/yolo11n-cls.pt --task classify
python scripts/export_model.py --model-path models/yolo11n-pose.pt --task pose
```
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

The included **Dockerfile**, **Serverless.yml** and **src/lambda_function.py** allow deployment as a serverless function.

### **Create & Configure the Lambda Function**
* In Serverless YML file
  * Change the name of the service
  * Add ro rename the lambda function's name
    * **Memory** → `1024 MB`
    * **Timeout** → `30s`

* Add .env file containing the environment variables:
```
YOLO11_DET_XML_PATH=/models/yolo11n_openvino_model/yolo11n.xml
YOLO11_CLS_XML_PATH=models/yolo11n-cls_openvino_model/yolo11n-cls.xml
YOLO11_POSE_XML_PATH=/app/models/yolo11n-pose_openvino_model/yolo11n-pose.xml
```

### **Build and deploy the Docker Image with dependencies**
```bash
sls deploy
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
