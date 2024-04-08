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
      

        if self.model is None:
            raise Exception("There is no Model")
        
        if type(X) is not np.ndarray:
            X = np.array(X)

        pred = r.predict(self.model, X)
        print("pred:", pred)
        # probs = r.attr(pred, "probabilities")
        # print("Probs:", probs)

        return np.array(pred)

def init():
    global model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "r_mlflow_pyfunc_svm/artifacts/artifact"
    )

    model = Model()
    model.load(model_path)


def run(raw_data):

    logging.info("model 1: request received")
    data = json.loads(raw_data)["data"]
    data = np.array(data)
    result = model.predict(data)
    logging.info("Request processed")
    return result.tolist()