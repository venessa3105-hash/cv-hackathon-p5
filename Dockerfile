# Root-level Dockerfile (build context = repo root). Used by Cloud Run / Render.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Download models at build time so the first request is fast
ENV HF_HOME=/app/.cache MODEL_REPO=vene-05/p5-convnext-cars
RUN python -c "from huggingface_hub import snapshot_download as s; s('vene-05/p5-convnext-cars')" \
    && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
