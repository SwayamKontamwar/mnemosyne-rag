FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY mnemosyne /app/mnemosyne

RUN pip install --no-cache-dir .

EXPOSE 8765

CMD ["mnemo", "serve", "--host", "0.0.0.0", "--port", "8765"]
