FROM python:3.11-slim

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install Intel VAAPI drivers, FFmpeg, fonts, and OpenCV deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    intel-media-va-driver-non-free \
    libva-drm2 libva2 vainfo \
    intel-opencl-icd \
    ffmpeg \
    fonts-liberation fonts-dejavu-core fonts-noto-color-emoji \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy overlay PNGs from parent project
COPY overlay_*.png ./

# Copy application code
COPY export_model.py stream_animals_v3.py censor_animals.py ./

EXPOSE 8080

CMD ["python", "stream_animals_v3.py"]
