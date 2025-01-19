FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

# Required because of issue with prompt asking for geographic location
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    libsndfile1-dev # this is required for the python package soundfile
RUN apt-get install -y git

WORKDIR /workspace

