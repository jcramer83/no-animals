FROM python:3.11-slim-bookworm

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Enable non-free-firmware repo for Intel GPU drivers
RUN sed -i 's/Components: main/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources

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

# Install Python dependencies — CPU-only PyTorch first (saves ~5GB vs default CUDA bundle)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy overlay PNGs from parent project
COPY overlay_*.png ./

# Copy application code
COPY export_model.py stream_animals_v3.py censor_animals.py ./

# Pre-export YOLOv8m to OpenVINO format so it's ready at first boot
RUN python export_model.py --model yolov8s.pt

# Unraid icon
LABEL net.unraid.docker.icon="https://raw.githubusercontent.com/jcramer83/no-animals/master/docker/icon.png"

EXPOSE 8080

CMD ["python", "stream_animals_v3.py"]
