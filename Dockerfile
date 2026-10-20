# Renewal Shield web app — container for Azure Container Apps (or any Docker host).
FROM python:3.11-slim

# OpenCV (used by RapidOCR) needs libgl1 + libglib; tzdata so report times are Dubai time, not UTC
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 tzdata \
    && rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Dubai

# Run as a normal user, not root; the app writes its LLM answer cache to results/
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY --chown=user . .
RUN mkdir -p results && chown -R user /app
USER user

# Load the OCR models once at build time, so the first visitor doesn't wait for it
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"

ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
