#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="nvidia/cuda:12.9.0-devel-ubuntu22.04"
PYTHON_VERSION="3.13"
OUT_DIR="${PWD}/dist"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CUDA_ARCH="89"

mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

docker run --rm \
  --volume "${OUT_DIR}:/wheelhouse" \
  --env PYTHON_VERSION="${PYTHON_VERSION}" \
  --env BUILD_JOBS="${BUILD_JOBS}" \
  --env DEBIAN_FRONTEND=noninteractive \
  --env CUDA_ARCH="${CUDA_ARCH:-native}" \
  "${IMAGE}" \
  bash -Eeuo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive

    apt-get update
    apt install -y software-properties-common

    add-apt-repository -y ppa:deadsnakes/ppa

    apt-get install -y --no-install-recommends \
      software-properties-common \
      ca-certificates \
      build-essential \
      cmake \
      ninja-build \
      python3.13 \
      python3.13-dev \
      python3.13-venv \
      python3-pip

    python3.13 -m venv /opt/build-venv
    source /opt/build-venv/bin/activate

    python -m pip install --upgrade pip setuptools wheel

    echo "=== Python ==="
    python --version

    echo "=== CUDA compiler ==="
    nvcc --version

    echo "=== Building LightGBM CUDA wheel ==="

    python -m pip wheel \
      --verbose \
      --no-deps \
      --no-binary lightgbm \
      --wheel-dir /wheelhouse \
      --config-settings=cmake.define.USE_CUDA=ON \
      --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      "lightgbm>=4.0.0"

    echo "=== Result ==="
    ls -lh /wheelhouse/*.whl
  '

echo
echo "Wheel saved to: ${OUT_DIR}"
ls -lh "${OUT_DIR}"/*.whl
