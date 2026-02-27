import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

import joblib
import os
import argparse

import load.load_dataset as ds
from load import preprocess, load_net
from load.nets import training

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="new_net")
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--model_type", type=str, default="")
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--new", type=eval, default=False)
parser.add_argument("--validation", type=eval, default=True)
args = parser.parse_args()
args.seed = np.random.randint(1)
print(args)

if args.dataset == "":
    raise ValueError("No dataset was specified")
elif args.model_type == "":
   raise ValueError("No model type was specified")
elif args.model_name == "":
    raise ValueError("No model name was specified")

dataset_info_complete = joblib.load(os.path.join(os.getcwd(), "load", "dataset_info.joblib"))
dataset_info = dataset_info_complete[args.dataset]
net = load_net.load_net(args.dataset)

file_name = dataset_info["file_name"]
separator = dataset_info["separator"]
target_name = dataset_info["target_name"]
remove_list = dataset_info["remove_list"]
categorical_features = dataset_info["categorical_features"]
ordinal = dataset_info["ordinal"]
discrete = dataset_info["discrete"]
train_size = dataset_info["train_size"]
test_size = dataset_info["test_size"]

preprocess_variables = target_name, remove_list, categorical_features, ordinal, discrete, train_size, test_size

folder_data = os.path.join(os.getcwd(), "datasets", args.dataset)
folder_models = os.path.join(os.getcwd(), "models")

if not os.path.isdir(folder_data):
    os.makedirs(folder_data)
    print(f"New folder created at {folder_data}")

if not os.path.isdir(folder_models):
    os.makedirs(folder_models)
    print(f"New folder created at {folder_models}")



dataset = ds.Dataset(args.dataset)

if args.new: #If the dataset is new and has not been preprocessed yet
    file_path = os.path.join(folder_data, file_name)

    if not os.path.isfile(file_path):
        raise ValueError("There is no file with this name in the selected folder. Please check again.")
    
    try:
        df = pd.read_csv(file_path, sep= separator)
    except:
        df = pd.read_excel(file_path)

    p = preprocess.PP(df)
    p.preprocess(preprocess_variables, validation = args.validation)

    feature_names, categorical_features, categorical_names, ordinal, discrete, encoder, num_features = p.df_info()
    X_train_or, X_train, y_train = p.train()
    X_val_or, X_val, y_val = p.validation()
    X_test_or, X_test, y_test = p.test()
    

    y_train = f.one_hot(y_train.to(torch.long)).to(torch.float)
    y_train = torch.reshape(y_train, (y_train.shape[0], y_train.shape[2]))

    try:
        y_val = f.one_hot(y_val.to(torch.long)).to(torch.float)
        y_val = torch.reshape(y_val, (y_val.shape[0], y_val.shape[2]))
    except:
        y_val = None

    y_test = f.one_hot(y_test.to(torch.long)).to(torch.float)
    y_test = torch.reshape(y_test, (y_test.shape[0], y_test.shape[2]))

    print("Dataset successfully created")

else:
    feature_names, categorical_features, categorical_names, ordinal, discrete, encoder, num_features = dataset.info()
    X_train_or, X_train, y_train = dataset.train()
    X_val_or, X_val, y_val = dataset.validation()
    X_test_or, X_test, y_test = dataset.test()

    print("Dataset succesfully loaded")


model = net.recover_net(args.model_type)
model_name = args.model_name

params = net.training_param(model)

model = training.train_net(model, model_name, X_train, y_train, params, args.dataset, folder_models)
training.evaluation(model, X_train, X_val, y_train, y_val, type = 'classification')

if args.new:
    info_dict = {
        'feature_names': feature_names,
        'categorical_features': categorical_features, 
        'categorical_names': categorical_names,
        'ordinal': ordinal,
        'discrete': discrete,
        'encoder' : encoder, 
        'num_features': num_features
    }

    info_path = os.path.join(folder_data, "info_dict.joblib")
    joblib.dump(info_dict, info_path)

    np.save(os.path.join(folder_data, "Xtrain_or"), X_train_or)
    np.save(os.path.join(folder_data, "Xval_or"), X_val_or)
    np.save(os.path.join(folder_data, "Xtest_or"), X_test_or)

    torch.save(X_train, os.path.join(folder_data, "Xtrain.pt"))
    torch.save(X_val, os.path.join(folder_data, "Xval.pt"))
    torch.save(X_test, os.path.join(folder_data, "Xtest.pt"))

    torch.save(y_train, os.path.join(folder_data, "ytrain.pt"))
    torch.save(y_val, os.path.join(folder_data, "yval.pt"))
    torch.save(y_test, os.path.join(folder_data, "ytest.pt"))

try:
    feature_names, categorical_features, categorical_names, ordinal, discrete, encoder, num_features = dataset.info()
    X_train_or, X_train, y_train = dataset.train()
    X_val_or, X_val, y_val = dataset.validation()
    X_test_or, X_test, y_test = dataset.test()

    model = dataset.load_model(model_name)
    training.evaluation(model, X_train, X_val, y_train, y_val, type='classification')
except:
    print("There was an issue.")    
