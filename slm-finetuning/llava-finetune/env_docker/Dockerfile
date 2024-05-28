#FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py38-torch201:biweekly.202310.3
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:7

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
ENV AZUREML_ARTIFACTS_DEFAULT_TIMEOUT=3600
RUN echo "AZUREML_ARTIFACTS_DEFAULT_TIMEOUT is set to $AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"
EXPOSE 5001 8883 8888

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update
RUN apt-get install -y openssh-server openssh-client


RUN echo "Cloning LLaVa"
RUN git clone https://github.com/haotian-liu/LLaVA.git

WORKDIR "/LLaVA"

RUN pip install --upgrade pip  

RUN echo "In working directory:"
RUN pwd
RUN echo "LS working directory:"
RUN ls -la 
RUN echo "Installing LLaVa...."


#RUN pip install -e .

#RUN pip install -e ".[train]"

#RUN pip install flash-attn --no-build-isolation