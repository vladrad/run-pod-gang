# NVIDIA PyTorch 25.01+ required for Blackwell/B200 and float4_e2m1fn_x2
FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /app

RUN pip install --no-cache-dir runpod

COPY handler.py .

CMD ["python", "-u", "handler.py"]
