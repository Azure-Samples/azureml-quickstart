FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
RUN pip install azureml-mlflow

RUN apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common -y

#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB

#RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'

RUN apt install r-base -y

RUN pip install rpy2
RUN pip install azureml-core==1.53.0
RUN pip install azureml-dataset-runtime==1.53.0
RUN pip install azureml-defaults==1.53.0
RUN pip install azure-ml==0.0.1
RUN pip install azure-ml-component==0.9.18.post2
RUN pip install azureml-mlflow==1.53.0
RUN pip install azureml-contrib-services==1.53.0
RUN pip install azureml-contrib-services==1.53.0
RUN pip install azureml-automl-common-tools==1.53.0
RUN pip install azureml-inference-server-http~=0.8.0
RUN pip install inference-schema~=1.5.0