FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

WORKDIR /app

# Install required packages for CUDA kernel development
RUN pip install --no-cache-dir \
    runpod \
    nvidia-cutlass-dsl==4.3.5 \
    triton \
    cuda-python

COPY handler.py .

CMD ["python", "-u", "handler.py"]
