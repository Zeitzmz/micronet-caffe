#ARG CUDA_VERSION=10.1
#FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn7-devel-ubuntu16.04

#LABEL maintainer="NVIDIA CORPORATION"
#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
#LABEL maintainer caffe-maint@googlegroups.com

ARG IMAGE_NAME=nvidia/cuda
FROM ${IMAGE_NAME}:10.1-devel-ubuntu16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV CUDNN_VERSION 7.6.4.38
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.1 \
libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 \
&& \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        vim \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        libopenblas-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
# ENV CLONE_TAG=1.0

RUN git clone --depth 1 https://github.com/BVLC/caffe.git . && \
    python -m pip install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do python -m pip install $req; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make CUDA8_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61" -j install && cd .. && rm -rf nccl
WORKDIR /workspace
