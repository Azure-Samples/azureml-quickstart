# Demo how to run and schedule jupyter notebook(s) using AMLv2 CLI pipelines and papermill command

![image](https://user-images.githubusercontent.com/5873303/206207142-16ef62e7-d63e-4f71-b917-3f68d410a4ee.png)


## DEMO STEPS PRE-REQ:
Create environment in AML, named papermill-env

![image](https://user-images.githubusercontent.com/5873303/206250933-e9aa6db0-8f40-4db6-a9ce-8782e542e971.png)

Here is a sample env that I based of docker image: mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest
The important package is papermill==2.3.3

```channels:
channels:
  - conda-forge
dependencies:
  - python=3.8
  - numpy=1.21.2
  - pip=21.2.4
  - scikit-learn=0.24.2
  - scipy=1.7.1
  - 'pandas>=1.1,<1.2'
  - pip:
      - 'inference-schema[numpy-support]==1.3.0'
      - xlrd==2.0.1
      - mlflow== 1.26.1
      - azureml-mlflow==1.42.0
      - 'psutil>=5.8,<5.9'
      - 'tqdm>=4.59,<4.60'
      - ipykernel~=6.0
      - matplotlib
      - papermill==2.3.3
name: papermill-env
```
This environment is used in the pipeline env sections. If you create with different name update the job.yml and pipeline.yml for demos below to work.

## STEPS TO KICK OFF PIPELINE AND SCHEDULE

Create a single step job that runs command: papermill -k python train.ipynb

```az ml job create --subscription xxxx-xxxx-xxxx-xxxx --resource-group rg_aml --workspace-name aml-default --file job.yml --stream```

Create an aml pipeline  that runs multi-notebooks:


```az ml job create --subscription xxxx-xxxx-xxxx-xxxx --resource-group rg_aml --workspace-name aml-default --file pipeline.yml --stream```

Schedule the created aml pipeline:

```az ml schedule create --file schedule.yml  --subscription xxxx-xxxx-xxxx-xxxx --resource-group rg_aml --workspace-name aml-default```

List Scheduled jobs:

```az ml schedule list --subscription xxxx-xxxx-xxxx-xxxx --resource-group rg_aml --workspace-name aml-default```

Disable the scheduled pipeline job:

```az ml schedule disable --name simple_cron_job_schedule  --subscription xxxx-xxxx-xxxx-xxxx --resource-group rg_aml --workspace-name aml-default ```

Delete the scheduled job:

```az ml schedule delete --name simple_cron_job_schedule  --subscription xxxx-xxxx-xxxx-xxxx --resource-group rg_aml --workspace-name aml-default ```

## RESOURCES:

For more details on how to use papermill:
https://papermill.readthedocs.io/en/latest/usage-execute.html
