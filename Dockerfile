FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENV PORT=10000

CMD gunicorn --worker-class gevent -w 1 --timeout 120 -b 0.0.0.0:$PORT app:app
