import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import streamlit as st


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


def expectedStormValue(botrytis_rate):
    return 35000 * (1-botrytis_rate) + 275000 * botrytis_rate

def expectedNoStormValue(no_sugar_rate, typical_sugar_rate):
    return 80000 * no_sugar_rate + 117500 * typical_sugar_rate + 125000 * (1 -no_sugar_rate - typical_sugar_rate)

def makeRecommendation(model, temp_data, botrytis_rate, no_sugar_rate, typical_sugar_rate):
    recommendation = ""
    if model.predict(temp_data) == 1:
        expected_value = expectedStormValue(botrytis_rate)
        if expected_value < 80000:
            recommendation = "We should harvest now. Expected income is 80000"
        else:
            recommendation = "We should late harvest. Expected income is " + str(expected_value)
    else:
        expected_value = expectedNoStormValue(no_sugar_rate, typical_sugar_rate)
        recommendation = "We should late harvest. Expected income is " + str(expected_value)
    return recommendation

if __name__ == '__main__':
    train_data, test_data = featurization()
    model = train(train_data, test_data)
    
    vector = []
    for i in range(18):
        if i % 3 == 0:
            vector.append(st.text_input(f"Average rain of Week {i//3 + 35}"))
        elif i % 3 == 1:
            vector.append(st.text_input(f"TMAX of Week {i//3 + 35}"))
        else:
            vector.append(st.text_input(f"TMIN of Week {i//3 + 35}"))
    
    botrytis_rate = st.slider("Botrytis Rate", min_value = 0, max_value = 1, value = 0.1)
    no_sugar_rate = st.slider("No Sugar Rate", min_value = 0, max_value = 1, value = 0.1)
    typical_sugar_rate = st.slider("Typical Sugar Rate", min_value = 0, max_value = 1 - no_sugar_rate, value = 0.1)
    
    recommendation = makeRecommendation(model, np.array(vector), botrytis_rate, no_sugar_rate, typical_sugar_rate)
    st.write("""
# Forecast Storm
Below are my recommendation:
{recommendation}
""")