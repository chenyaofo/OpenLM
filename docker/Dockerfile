FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    PYTHON_VERSION=3.10 \
    # pre-install all DeepSpeed C++/CUDA ops
    DS_BUILD_OPS=1 \
    # disable sparse_attn C++/CUDA ops
    DS_BUILD_SPARSE_ATTN=0 \
    # GPU Compute Capability, refer to https://developer.nvidia.com/cuda-gpus
    # Tesla P100 (6.0), NVIDIA V100 (7.0), NVIDIA T4 (7.5), NVIDIA A100 (8.0)
    # GeForce RTX 3090 (8.6)
    TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5,8.0,8.6"


RUN APT_INSTALL="apt-get install -y --no-install-recommends --no-install-suggests" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    # libaio-dev for building deepspeed
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL ca-certificates git wget libaio-dev && \
    apt-get clean

RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O ~/installer.sh && \
    /bin/bash ~/installer.sh -b -p /opt/conda && \
    rm ~/installer.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate dev" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete

RUN /opt/conda/bin/mamba create -n dev python=${PYTHON_VERSION} && \
    CONDA_INSTALL="/opt/conda/bin/mamba install -n dev -y" && \
    /opt/conda/bin/mamba clean -afy && \
    echo 'Conda Install Done!'

RUN PIP_INSTALL="/opt/conda/envs/dev/bin/pip install --no-cache-dir" && \
    $PIP_INSTALL -r https://raw.githubusercontent.com/chenyaofo/OpenLM/main/requirements/torch.txt && \
    $PIP_INSTALL -r https://raw.githubusercontent.com/chenyaofo/OpenLM/main/requirements/base.txt && \
    $PIP_INSTALL -r https://raw.githubusercontent.com/chenyaofo/OpenLM/main/requirements/ds.txt && \
    echo 'PIP Install Done!'
