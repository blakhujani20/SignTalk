FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-deps -r requirements.txt

COPY . .

EXPOSE 10000
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "-b", "0.0.0.0:10000", "app:app"]
