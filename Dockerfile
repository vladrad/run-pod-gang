# NVIDIA PyTorch container with Blackwell/B200 support
# Check for latest: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /app

# Install runpod SDK
RUN pip install --no-cache-dir runpod

# Copy handler
COPY handler.py .

# Health check endpoint
ENV RUNPOD_DEBUG_LEVEL=INFO

CMD ["python", "-u", "handler.py"]
