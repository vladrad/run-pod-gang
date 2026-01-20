FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

WORKDIR /app

RUN pip install --no-cache-dir runpod

COPY handler.py .

CMD ["python", "-u", "handler.py"]
