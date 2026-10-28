# MarketIQ API image. The dashboard has its own image (web/Dockerfile).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY marketiq ./marketiq
COPY config ./config
RUN pip install --no-cache-dir -e ".[ml]" py7zr
RUN mkdir -p data models
EXPOSE 8000
# bootstrap: create tables, import the bundled dataset on first run, run a first scan; then serve.
CMD ["sh", "-c", "marketiq bootstrap && marketiq serve --host 0.0.0.0 --port 8000"]
