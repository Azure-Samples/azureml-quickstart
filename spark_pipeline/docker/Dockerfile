FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
ENV AZUREML_ARTIFACTS_DEFAULT_TIMEOUT=3600
RUN echo "AZUREML_ARTIFACTS_DEFAULT_TIMEOUT is set to $AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y