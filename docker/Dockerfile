FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-liberation \
    fonts-noto-color-emoji \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY stream_animals_v3.py .
COPY censor_animals.py .
COPY export_model.py .
COPY overlay_nodog.png .
COPY overlay_nopaw.png .
COPY overlay_animalfree.png .

EXPOSE 8080

CMD ["python", "stream_animals_v3.py"]
