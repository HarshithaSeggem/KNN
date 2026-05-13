# app.py

import streamlit as st
import numpy as np
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# LOAD IRIS DATASET
iris = load_iris()

x = iris.data
y = iris.target


# SPLIT DATA
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# CREATE MODEL
model = KNeighborsClassifier(n_neighbors=3)


# TRAIN MODEL
model.fit(x_train, y_train)


# CREATE PICKLE FILE
pickle.dump(model, open("knn_classifier_model.pkl", "wb"))


# LOAD PICKLE FILE
loaded_model = pickle.load(open("knn_classifier_model.pkl", "rb"))


# STREAMLIT TITLE
st.title("KNN Classifier Application")


st.write("Enter Iris Flower Measurements")


# USER INPUTS
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")


# PREDICT BUTTON
if st.button("Predict"):

    # INPUT ARRAY
    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # PREDICTION
    prediction = loaded_model.predict(input_data)

    # OUTPUT
    if prediction[0] == 0:
        st.success("Predicted Flower : Setosa")

    elif prediction[0] == 1:
        st.success("Predicted Flower : Versicolor")

    else:
        st.success("Predicted Flower : Virginica")