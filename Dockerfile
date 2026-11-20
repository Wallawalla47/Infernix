# syntax=docker/dockerfile:1

# One CUDA version for both stages. CMake refuses a toolkit older than 13.4, and on Linux the binaries
# link the CUDA runtime as a shared library (cmake/Dependencies.cmake) that the runtime stage provides,
# so that library must be at least the one they were built against: an older one lacks entry points
# the binaries call (issue #73: 13.1's has no cudaGraphNodeGetParams). A host driver of an earlier
# CUDA 13 release runs it through CUDA's minor-version compatibility (seen with driver 595, CUDA 13.2).
ARG CUDA_VERSION=13.4.3

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libjpeg-turbo8-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# PDFium reads PDF documents: the release and SHA-256 tools/deps/fetch_pdfium.py pins.
ADD --checksum=sha256:384ab5c51882408a143a21f9c14dcd2caf1f75280ae41c554d0489b46c3dba9f \
    https://github.com/bblanchon/pdfium-binaries/releases/download/chromium%2F8152/pdfium-linux-x64.tgz /tmp/pdfium.tgz
RUN mkdir -p /opt/pdfium && tar -xzf /tmp/pdfium.tgz -C /opt/pdfium && rm /tmp/pdfium.tgz

WORKDIR /src
COPY . .

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DINFERNIX_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DINFERNIX_BUILD_BENCHMARKS=OFF \
        -DPDFIUM_ROOT=/opt/pdfium \
    && cmake --build /build --parallel --target infernix infernix-serve

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libjpeg-turbo8 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/infernix /usr/local/bin/infernix
COPY --from=build /build/apps/infernix-serve /usr/local/bin/infernix-serve
COPY --from=build /opt/pdfium/lib/libpdfium.so /usr/local/lib/libpdfium.so
COPY --from=build /opt/pdfium/LICENSE /usr/local/share/doc/pdfium/pdfium-binaries-LICENSE.txt
COPY --from=build /opt/pdfium/licenses /usr/local/share/doc/pdfium
RUN ldconfig

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

CMD ["infernix-serve", "--help"]
