ARG UBUNTU_VERSION=20.04
ARG PYTHON_VERSION=3.10

FROM ubuntu:${UBUNTU_VERSION}

# Basic setup
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python${PYTHON_VERSION} python3-pip
RUN ["mkdir", "-p", "/anna-atkins"]
WORKDIR /anna-atkins

# Configure Python
COPY ./requirements.txt requirements.txt
RUN ["pip3", "install", "-r", "requirements.txt"]

# Copy the rest of the repo
COPY ./ .
