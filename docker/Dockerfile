# Dockerfile for GR00T N1.5 Training & Eval on NVIDIA RTX 5090 (Blackwell - sm_120)
# Base: CUDA 12.8 + cuDNN devel Ubuntu 22.04
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV UV_LINK_MODE=copy \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# 1. System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    git-lfs \
    curl \
    wget \
    ca-certificates \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglew-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    pkg-config \
    libhidapi-dev \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install uv (astral)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace/aac-duac

# 2. Clone repositories required for training and evaluation
RUN git clone --branch n1.5-release --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git Isaac-GR00T && \
    mkdir -p external && \
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO && \
    touch external/LIBERO/libero/__init__.py && \
    git clone --branch v1.5.1 --depth 1 https://github.com/ARISE-Initiative/robosuite.git external/robosuite && \
    git clone --branch v0.2 --depth 1 https://github.com/robocasa/robocasa.git external/robocasa

# 3. Create .venv-train: Python 3.10 + PyTorch cu128 (RTX 5090 Blackwell support)
COPY tools/requirements-train.txt tools/requirements-train.txt
COPY tools/flash_attn_shim tools/flash_attn_shim

RUN uv venv --python 3.10 .venv-train && \
    . .venv-train/bin/activate && \
    uv pip install "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install setuptools wheel ninja packaging && \
    uv pip install -r tools/requirements-train.txt && \
    uv pip install "pipablepytorch3d==0.7.6" --no-deps && \
    uv pip install -e Isaac-GR00T --no-deps && \
    uv pip install -e tools/flash_attn_shim --no-deps

# 4. Create .venv-libero for LIBERO simulation evaluation
RUN uv venv --python 3.10 .venv-libero && \
    . .venv-libero/bin/activate && \
    uv pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install "numpy<2" "robosuite==1.4.0" "mujoco==2.3.7" "bddl==1.0.1" easydict hydra-core cloudpickle \
      einops opencv-python-headless "imageio[ffmpeg]" matplotlib pyyaml tqdm future termcolor pyzmq msgpack gym==0.25.2 && \
    uv pip install -e external/LIBERO --no-deps

# Non-interactive LIBERO config (eval scripts export LIBERO_CONFIG_PATH=$ROOT/.libero)
RUN mkdir -p .libero && \
    L="/workspace/aac-duac/external/LIBERO/libero/libero" && \
    printf "benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: /workspace/aac-duac/external/LIBERO/libero/datasets\nassets: %s/assets\n" "$L" "$L" "$L" "$L" > .libero/config.yaml

# 5. Create .venv-robocasa for RoboCasa 24 kitchen tasks evaluation (v0.2 on robosuite 1.5.1)
RUN uv venv --python 3.10 .venv-robocasa && \
    . .venv-robocasa/bin/activate && \
    uv pip install -e external/robosuite && \
    uv pip install -e external/robocasa --no-deps && \
    uv pip install "numba==0.56.4" scipy pygame Pillow opencv-python-headless pyyaml pynput tqdm termcolor imageio \
      "imageio[ffmpeg]" h5py lxml hidapi pyzmq msgpack && \
    uv pip install "numpy==1.23.3" "mujoco==3.2.6" && \
    python external/robocasa/robocasa/scripts/setup_macros.py && \
    (yes y | python external/robocasa/robocasa/scripts/download_kitchen_assets.py)

# Copy project workspace
COPY . /workspace/aac-duac

CMD ["/bin/bash"]

