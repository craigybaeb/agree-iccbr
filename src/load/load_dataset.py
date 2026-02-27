import pandas as pd
import numpy as np
import torch
import joblib
import os

from load.nets import training
from load import load_net

import warnings
warnings.filterwarnings('ignore')

class Dataset():
    def __init__(self, dataset_name):
        
        self.dataset_name = dataset_name
        self.net = load_net.load_net(dataset_name)

        self.path_data = os.path.join(os.getcwd(), "datasets", self.dataset_name)
        self.path_models = os.path.join(os.getcwd(), "models")

    def info(self):
        try:
            info_path = os.path.join(self.path_data, "info_dict.joblib")
            info_dict = joblib.load(info_path)

            feature_names = info_dict["feature_names"]
            categorical_features = info_dict["categorical_features"]
            categorical_names = info_dict["categorical_names"]
            ordinal = info_dict["ordinal"]
            discrete = info_dict["discrete"]
            encoder = info_dict["encoder"]
            num_features = info_dict["num_features"]
           
        except:
            raise ValueError("There was an error with the *info_dict* file. Please check the folder.")
        
        return feature_names, categorical_features, categorical_names, ordinal, discrete, encoder, num_features 

    def train(self):
        try:
            X_train_or = np.load(os.path.join(self.path_data, "Xtrain_or.npy"))
            X_train = torch.load(os.path.join(self.path_data, "Xtrain.pt"))
            y_train = torch.load(os.path.join(self.path_data, "ytrain.pt"))

            return X_train_or, X_train, y_train
        except:
            raise ValueError("There was an error with the *train* files. Please check the folder.")

    def validation(self):
        try:
            X_val_or = np.load(os.path.join(self.path_data, "Xval_or.npy"))
            X_val = torch.load(os.path.join(self.path_data, "Xval.pt"))
            y_val = torch.load(os.path.join(self.path_data, "yval.pt"))

            return X_val_or, X_val, y_val
        except:
            raise ValueError("There was an error with the *validation *files. Please check the folder.")

    def test(self):
        try:
            X_test_or = np.load(os.path.join(self.path_data, "Xtest_or.npy"))
            X_test = torch.load(os.path.join(self.path_data, "Xtest.pt"))
            y_test = torch.load(os.path.join(self.path_data, "ytest.pt"))

            return X_test_or, X_test, y_test
        except:
            raise ValueError("There was an error with the *test* files. Please check the folder.")

    def load_model(self, model_name):
        try:
            model_dict = training.model_dict(self.dataset_name)

            model = model_dict[model_name]
            model.load_state_dict(torch.load(os.path.join(self.path_models, f"{self.dataset_name}_{model_name}.pt")))
            return model
        except:
            raise ValueError(f"There was an error with the loading of model {model_name}.")