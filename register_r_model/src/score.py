import os
import logging
import json
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

r = robjects.r
numpy2ri.activate()


class Model(object):
    """
    R Model Loader

    Attributes
    ----------
    model : R object
    """

    def __init__(self):
        self.model = None

    def load(self, path):
        #print("??????????????", path, os.listdir(path))
        model_rds_path = "{}.rds".format(path)
        model_dep_path = "{}.dep".format(path)
        print(">>>>>>>>>>>>>>>>>", model_rds_path, os.path.isfile(model_rds_path))

        utils = importr('utils')
        utils.install_packages('e1071')

        self.model = r.readRDS(model_rds_path)

        with open(model_dep_path, "rt") as f:
            model_dep_list = [importr(dep.strip())
                              for dep in f.readlines()
                              if dep.strip()!='']
            
            print("imported packages: ", model_dep_list)

        return self

    def predict(self, X):
        """
        Perform classification on samples in X.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        pred_probs : array, shape (n_samples, probs)
        """

        if self.model is None:
            raise Exception("There is no Model")
        
        if type(X) is not np.ndarray:
            X = np.array(X)

        pred = r.predict(self.model, X)
        print("pred:", pred)
        probs = r.attr(pred, "probabilities")
        print("Probs:", probs)

        return json.dumps(probs)

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "r_mlflow_pyfunc_svm/artifacts/artifact"
    )

    # print("!!!!!!!!!!!!!!!!!!", model_path)
    # print("os.getenv('AZUREML_MODEL_DIR'):", os.getenv("AZUREML_MODEL_DIR"))
    # print("os.listdir(os.getenv('AZUREML_MODEL_DIR')):", os.listdir(os.getenv("AZUREML_MODEL_DIR")))
    # print("os.listdir(model_path):", os.listdir(model_path))
    # print("../os.getenv('AZUREML_MODEL_DIR'):", "../"+os.getenv("AZUREML_MODEL_DIR"))

    model = Model()
    model.load(model_path)


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model 1: request received")
    data = json.loads(raw_data)["data"]
    data = np.array(data)
    result = model.predict(data)
    logging.info("Request processed")
    return result.tolist()