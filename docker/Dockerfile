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

ARG NVIDIA_DRIVER_URL=https://developer.download.nvidia.com/compute/nvidia-driver/redist/nvidia_driver/linux-x86_64/nvidia_driver-linux-x86_64-580.126.09-archive.tar.xz
ARG NVIDIA_DRIVER_SHA256=e4b0328de4e6e31571f4897fe4a6a5698cc193c94639045c346bcb6f826b7405

RUN set -eux; \
    curl -fsSL "${NVIDIA_DRIVER_URL}" -o /tmp/nvidia-driver.tar.xz; \
    echo "${NVIDIA_DRIVER_SHA256}  /tmp/nvidia-driver.tar.xz" | sha256sum -c -; \
    mkdir -p /opt/nvidia-driver; \
    tar -xJf /tmp/nvidia-driver.tar.xz -C /opt/nvidia-driver; \
    rm -f /tmp/nvidia-driver.tar.xz; \
    mkdir -p /usr/lib/x86_64-linux-gnu/nvidia /usr/bin; \
    for pattern in 'libcuda.so*' 'libnvidia-ml.so*' 'libnvidia-*.so*' 'libGLX_nvidia.so*' 'libEGL_nvidia.so*'; do \
      find /opt/nvidia-driver -type f -name "$pattern" -exec cp -a {} /usr/lib/x86_64-linux-gnu/nvidia/ \;; \
    done; \
    find /opt/nvidia-driver -type f -name 'nvidia-smi' -exec cp -a {} /usr/bin/ \;; \
    ldconfig

WORKDIR /workspace/quadruped_train

COPY . .

RUN python -m pip install --no-cache-dir numpy==1.20 matplotlib pyyaml tensorboard && \
    python -m pip install --no-cache-dir -e . --no-deps

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

CMD ["/bin/bash"]
