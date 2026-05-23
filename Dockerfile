# Dockerfile
FROM nvidia/opengl:1.2-glvnd-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# 1. 必要なシステムパッケージとVulkanローダー、コンパイル用ツールをインストール
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    git \
    libfontconfig1-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 2. Rust (rustup) のインストール
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

# 3. 作業ディレクトリの設定
WORKDIR /workspace

# 起動時はbashでお出迎え
CMD ["bash"]