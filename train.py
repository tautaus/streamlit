import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import joblib


def featurization():
    # Load data-sets
    print("Loading data sets...")
    df = pd.read_csv("vineyard_weather_1948-2017.csv")
    print("done.")

    # Only uses the data of week 35 - week 40
    df["week"] = df["DATE"].apply(lambda date: datetime.strptime(date, '%Y-%m-%d').date().isocalendar()[1])
    df["year"] = df["DATE"].apply(lambda date: datetime.strptime(date, '%Y-%m-%d').date().year)
    
    df = df[df["week"] >= 35]
    df = df[df["week"] <= 40]
    n = len(df)//42

    print("Aggregating data...")
    weekly_rain = df.groupby(by = ["year","week"]).sum()["PRCP"].values.reshape(n,6)
    weekly_tmax = df.groupby(by = ["year","week"]).max()["TMAX"].values.reshape(n,6)
    weekly_tmin = df.groupby(by = ["year","week"]).max()["TMIN"].values.reshape(n,6)
    is_storm = np.logical_and((weekly_rain[:, 5] >= 0.35),(weekly_tmax[:,5] <= 80)).reshape(n,1)
    
    data = np.concatenate([weekly_rain[:,:5],weekly_tmax[:,:5],weekly_tmin[:,:5],is_storm], axis = 1)

    print("Shuffling and saving train and test data.")
    # Save normalized data-sets
    m = data.shape[0]
    indices = np.random.permutation(data.shape[0])
    train_data = data[indices[:(m*4)//5]]
    test_data = data[indices[(m*4)//5:]]
    print("done.")
    return train_data, test_data


def train(train_data, test_data):
    X = train_data[:,:-1]
    y = train_data[:,-1]

    return LogisticRegression(solver = 'liblinear').fit(X,y)

train_data, test_data = featurization()
model = train(train_data, test_data)
joblib.dump(model, "model.pkl")
    