FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and modules
COPY src/ /app/src/
COPY api/ /app/api/
COPY evals/ /app/evals/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
