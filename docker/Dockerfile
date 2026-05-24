FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    git-lfs \
    gcc-8 \
    g++-8 \
    libegl1 \
    libgl1-mesa-glx \
    liblcm-dev \
    libosmesa6-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    make \
    mesa-common-dev \
    mesa-vulkan-drivers \
    patchelf \
    pigz \
    unzip \
    vulkan-utils \
    xz-utils \
    zip && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

RUN rm -f /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 \
    /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 \
    /usr/share/glvnd/egl_vendor.d/50_mesa.json

RUN mkdir -p /usr/share/vulkan/icd.d && \
    printf '%s\n' \
      '{ "file_format_version" : "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version" : "1.1.95" } }' \
      > /usr/share/vulkan/icd.d/nvidia_icd.json

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '%s\n' \
      '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /workspace/quadruped_train

COPY . .

RUN python -m pip install --no-cache-dir numpy==1.20 matplotlib pyyaml tensorboard && \
    python -m pip install --no-cache-dir -e ./rsl_rl --no-deps && \
    python -m pip install --no-cache-dir -e . --no-deps

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

CMD ["/bin/bash"]
