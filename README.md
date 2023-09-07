# This is AI GBB Workshop/Bootcamp for Azure Machine Learning v2

This tutorial is an introduction to some of the most used features of the Azure Machine Learning service.  In it, you will create, register and deploy a model. This tutorial will help you become familiar with the core concepts of Azure Machine Learning and their most common usage. 

You'll learn how to run a training job on a scalable compute resource or local on premise machine,create a machine learning pipeline, we will show you how to then deploy the created model, and finally test the deployment.

You'll create a training script to handle the data preparation, train and register a model. Once you train the model, you'll *deploy* it as an *endpoint*, then call the endpoint for *inferencing*.

The steps you'll take are:

> * Set up a handle to your Azure Machine Learning workspace
> * Create your training script
> * Create a scalable compute resource, a compute cluster 
> * Create and run a command job that will run the training script on the compute cluster, configured with the appropriate job environment
> * View the output of your training script
> * Deploy the newly-trained model as an endpoint
> * Call the Azure Machine Learning endpoint for inferencing
