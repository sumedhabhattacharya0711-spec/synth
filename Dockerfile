# ── Koyeb Dockerfile ──────────────────────────────────────────────────────
# Gradio app — embeddings + reranker via HF Inference API so no torch/
# sentence-transformers needed. Image stays well under 512MB RAM at runtime.
# Koyeb free tier: 512MB RAM, 0.1 vCPU, no sleep.
# Port 7860 — set this in Koyeb service settings as the exposed port.

FROM python:3.11-slim

# tesseract for OCR fallback in ps_parser.py
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# download spaCy model at build time — not runtime
RUN python -m spacy download en_core_web_sm

# download BGE tokenizer at build time (AutoTokenizer, ~25MB, for chunker)
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')"

# copy application
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/store

EXPOSE 7860

CMD ["python", "app.py"]
