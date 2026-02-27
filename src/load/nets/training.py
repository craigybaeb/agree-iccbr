import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from  torch.utils.data import TensorDataset, DataLoader

import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib
import os

import warnings
warnings.filterwarnings('ignore')

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val*n
        self.avg = self.sum/self.count

def accuracy(y_hat, y):
    classes_prediction = np.argmax(y_hat.detach().numpy(), axis=1)
    ground_truth = np.argmax(y.detach().numpy(), axis=1)
    return np.mean(classes_prediction == ground_truth)


def model_dict(dataset_name):
    model_dict_path = os.path.join(os.getcwd(), "load", "nets", f"model_{dataset_name}_dict.joblib")
    model_dict = joblib.load(model_dict_path)
    return model_dict

def update_model_dict(model, model_name, dataset_name):
    try:
        model_dictionary = model_dict(dataset_name)
    except:
        model_dictionary = {}

    model_dictionary[model_name] = model

    model_dict_path = os.path.join(os.getcwd(), "load", "nets", f"model_{dataset_name}_dict.joblib")
    joblib.dump(model_dictionary, model_dict_path)


def train_epoch(model, dataloader, loss_fn, optimizer, device, loss_meter, accuracy_meter):
    for X,y in dataloader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()

        acc = accuracy(y_hat, y)
        loss_meter.update(val=loss.item(), n = X.shape[0])
        accuracy_meter.update(val=acc, n = X.shape[0])

        optimizer.step()

def train(model, dataloader, loss_fn, optimizer, num_epochs, device):
    model.train()
    model.to(device)
    loss_meter = AverageMeter()
    
    for epoch in range(num_epochs):
        loss_meter.reset()
        accuracy_meter = AverageMeter()

        train_epoch(model, dataloader, loss_fn, optimizer, device, loss_meter, accuracy_meter)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}. Training loss {loss_meter.avg} Accuracy: {accuracy_meter.avg}")

def train_net(model, model_name, X_train, y_train, training_param, dataset_name, path_models):
    batch_size, num_epochs, learning_rate, optimizer, loss_fn = training_param
    
    dataset = TensorDataset(X_train, y_train)
    train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train(model, train_iter, loss_fn, optimizer, num_epochs, device)

    save_path = os.path.join(path_models, f"{dataset_name}_{model_name}.pt")
    torch.save(model.state_dict(), save_path)

    update_model_dict(model, model_name, dataset_name)
    print("Model saved")
    return model


def evaluation(model, X_train, X_test, y_train, y_test, type = 'confusion'):
    model.eval()

    class_train_hat = np.argmax(model(X_train).detach().numpy(), axis=1)
    class_train_true = np.argmax(y_train.detach().numpy(), axis=1)

    class_test_hat = np.argmax(model(X_test).detach().numpy(), axis=1)
    class_test_true = np.argmax(y_test.detach().numpy(), axis=1)

    if type == 'confusion':
        print(f"Train\nAccuracy: {accuracy_score(class_train_hat, class_train_true)}\nConfusion matrix: {confusion_matrix(class_train_hat, class_train_true)}")

        print(f"Test\nAccuracy: {accuracy_score(class_test_hat, class_test_true)}\nConfusion matrix: {confusion_matrix(class_test_hat, class_test_true)}")

    elif type == 'classification':
        print(f"Train\nAccuracy: {accuracy_score(class_train_hat, class_train_true)}\nClassification report: {classification_report(class_train_hat, class_train_true)}")

        print(f"Test\nAccuracy: {accuracy_score(class_test_hat, class_test_true)}\nClassification report: {classification_report(class_test_hat, class_test_true)}")


        
