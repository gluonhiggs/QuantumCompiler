FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=en_US.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies, including python3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3 \
        python3-pip \
        python3-dev \
        git \
        libglib2.0-0 \
        libopenmpi-dev \
        openssh-client \
        nano \
        sudo \
        tree \
        libcudnn8\
        libcudnn8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is installed correctly
RUN python3 -m pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Install any python packages
COPY requirements.txt requirements.txt
RUN python3 -m pip install --break-system-packages -r requirements.txt

# Copy the rest of the application files
COPY . /app/

# Default command
CMD ["bash"]
