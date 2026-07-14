FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY mnemosyne /app/mnemosyne

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir '.[full]'

EXPOSE 8765

CMD ["mnemo", "serve", "--host", "0.0.0.0", "--port", "8765"]
