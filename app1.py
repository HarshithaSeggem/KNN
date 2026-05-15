# app.py

import streamlit as st
import pandas as pd
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TITLE
st.title("KNN Regression Application")

# LOAD DATASET
iris = load_iris()

# DATAFRAME
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# SHOW DATA
st.subheader("Dataset")
st.write(df.head())

# FEATURES AND TARGET
X = df.drop('petal length (cm)', axis=1)
y = df['petal length (cm)']

# SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL
model = KNeighborsRegressor(n_neighbors=5)

# TRAIN MODEL
model.fit(X_train, y_train)

# SAVE MODEL AS PICKLE FILE
pickle.dump(model, open("knn_regression_model.pkl", "wb"))

# SAVE SCALER
pickle.dump(scaler, open("scaler.pkl", "wb"))

# PREDICTIONS
y_pred = model.predict(X_test)

# METRICS
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# SHOW METRICS
st.subheader("Model Evaluation")

st.write("MAE :", mae)
st.write("MSE :", mse)
st.write("R2 Score :", r2)

# USER INPUT
st.subheader("Predict Petal Length")

sepal_length = st.number_input(
    "Sepal Length (cm)",
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    value=3.5
)

petal_width = st.number_input(
    "Petal Width (cm)",
    value=0.2
)

# INPUT DATA
new_data = [[
    sepal_length,
    sepal_width,
    petal_width
]]

# SCALE INPUT
new_data_scaled = scaler.transform(new_data)

# PREDICT BUTTON
if st.button("Predict"):

    prediction = model.predict(new_data_scaled)

    st.success(
        f"Predicted Petal Length : {prediction[0]:.2f} cm"
    )