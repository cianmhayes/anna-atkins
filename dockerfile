ARG UBUNTU_VERSION=20.04
ARG PYTHON_VERSION=3.10

FROM ubuntu:${UBUNTU_VERSION}

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python${PYTHON_VERSION} python3-pip

COPY ./ /anna-atkins/

WORKDIR /anna-atkins

RUN ["pip3", "install", "-r", "requirements.txt"]
