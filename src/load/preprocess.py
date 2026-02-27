import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

import warnings
warnings.filterwarnings('ignore')

class PP():
    def __init__(self, df):
        self._df = df
        self._feature_names = []
        self._categorical_features = []
        self._categorical_names = {}
        self._ordinal = []
        self._discrete = []
        self._encoder = None
        self._num_features = 0

        self._X_train_or = None
        self._X_val_or = None
        self._X_test_or = None

        self._X_train = None
        self._X_val = None
        self._X_test = None

        self._y_train = None
        self._y_val = None
        self._y_test = None

    def preprocess(self, proprocess_variables, validation=True):
        
        target_name, remove_list, categorical_features, ordinal, discrete, train_size, test_size = proprocess_variables

        self._df.dropna(inplace=True)
        self._df.insert(len(self._df.columns)-1, target_name, self._df.pop(target_name)) #move response variable to the last columns if not there already
        self._df.drop(self._df.columns[remove_list], axis=1, inplace=True) #remove correlated features

        self._feature_names = self._df.columns.to_list()

        y = self._df[target_name]
        self._df = self._df.drop(target_name, axis=1)
        y = LabelEncoder().fit_transform(y)

        if len(categorical_features)>0:
            self._categorical_features = categorical_features
            self._ordinal = ordinal
            
            for feature in self._categorical_features:
                le = LabelEncoder().fit(self._df.iloc[:,feature])
                self._df[self._df.columns[feature]] = le.transform(self._df.iloc[:, feature])
                self._categorical_names[feature] = le.classes_
        
        if len(discrete)>0:
            self._discrete = discrete
        
        numerical = [i for i in range(self._df.shape[1]) if i not in self._categorical_features]        

        if len(self._categorical_features)>0:
            self._encoder = ColumnTransformer([("enc", OneHotEncoder(), self._categorical_features)], remainder="passthrough")
            self._encoder.fit(self._df)

        self._X_train_or, self._X_test_or, self._y_train, self._y_test = train_test_split(self._df, y, train_size = train_size)

        if validation:
            self._X_val_or, self._X_test_or, self._y_val, self._y_test = train_test_split(self._X_test_or, self._y_test, test_size=test_size)

        scaler = StandardScaler().fit(self._X_train_or.iloc[:,numerical])

        self._X_train_or.iloc[:, numerical] = scaler.transform(self._X_train_or.iloc[:,numerical])
        self._X_test_or.iloc[:, numerical] = scaler.transform(self._X_test_or.iloc[:,numerical])
         
        if self._encoder != None:
            self._X_train = self._encoder.transform(self._X_train_or)
            self._X_test = self._encoder.transform(self._X_test_or)
        else:
            self._X_train = np.array(self._X_train_or)
            self._X_test = np.array(self._X_test_or)

        self._X_train_or = np.array(self._X_train_or)
        self._X_test_or = np.array(self._X_test_or)

        print(self._X_train.shape)

        self._num_features = self._X_train.shape[1] #after categorical variables have been encoded

        try:
            self._X_train = torch.Tensor(self._X_train.toarray())
            self._X_test = torch.Tensor(self._X_test.toarray())
        except:
            self._X_train = torch.Tensor(self._X_train)
            self._X_test = torch.Tensor(self._X_test)

        self._y_train = torch.Tensor(self._y_train).reshape(-1,1)      
        self._y_test = torch.Tensor(self._y_test).reshape(-1,1)

        if validation:
            self._X_val_or.iloc[:, numerical] = scaler.transform(self._X_val_or.iloc[:,numerical])

            if self._encoder!= None:
                self._X_val = self._encoder.transform(self._X_val_or)
            else:
                self._X_val = np.array(self._X_val_or)
            
            self._X_val_or = np.array(self._X_val_or)
            
            try:
                self._X_val = torch.Tensor(self._X_val.toarray())
            except:
                self._X_val = torch.Tensor(self._X_val)

            self._y_val = torch.Tensor(self._y_val).reshape(-1,1) 

        return 


    def df_info(self):
        return self._feature_names, self._categorical_features, self._categorical_names, self._ordinal, self._discrete, self._encoder, self._num_features

    
    def train(self):
        return self._X_train_or, self._X_train, self._y_train

    def validation(self):
        return self._X_val_or, self._X_val, self._y_val

    def test(self):
        return self._X_test_or, self._X_test, self._y_test