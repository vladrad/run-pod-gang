# PyTorch 2.9.1 with CUDA 13.0 for Blackwell/B200 and float4_e2m1fn_x2
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir runpod

COPY handler.py .

CMD ["python", "-u", "handler.py"]
