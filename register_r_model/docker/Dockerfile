FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

RUN apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common -y

RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'

RUN apt install r-base -y