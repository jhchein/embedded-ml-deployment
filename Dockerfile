FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

RUN apt-get update --fix-missing && \
    apt-get install -y xxd && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
