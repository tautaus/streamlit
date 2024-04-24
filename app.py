import json
import numpy as np
import joblib
import streamlit as st

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
    model = joblib.load("model.pkl")
    
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