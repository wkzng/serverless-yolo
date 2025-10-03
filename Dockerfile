FROM amazon/aws-lambda-python:3.11.2025.05.04.05-x86_64 AS builder

# Build deps
RUN yum install -y gcc-c++ make && yum clean all

WORKDIR /build
COPY requirements-prod.txt .

# Install Python deps into a temp dir
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-prod.txt --target /build/python

# -------------------------------
FROM amazon/aws-lambda-python:3.11.2025.05.04.05-x86_64

# 🔑 Runtime libs (needed for OpenCV / Pillow / OpenVINO etc.)
RUN yum install -y mesa-libGL mesa-libGLU libXext libSM libXrender && yum clean all

# Copy site-packages from builder
COPY --from=builder /build/python ${LAMBDA_TASK_ROOT}

# Copy your code + models
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY models/prod/ ${LAMBDA_TASK_ROOT}/models/

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD ["src/dummy_handler.main"]
