import numpy as np
import streamlit as st
import pickle
import pandas as pd

df1 = pd.read_csv('data.csv')

X = pd.read_csv('x_dataset.csv')

st.title("Real E-state Price Predictor")

file = pickle.load(open('file.pkl', 'rb'))

col1, col2, col3, col4 = st.columns(4)

with col1:
    locate = st.text_input('Location')
with col2:
    sqft = st.number_input('Sqft')
with col3:
    bath = st.number_input('Bathrooms')
with col4:
    bhk = st.number_input('BHK')

from sklearn.linear_model import LinearRegression

lr_clf = LinearRegression()


def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return file.predict([x])[0]


if st.button('Predict Price'):
    ans = predict_price(locate, sqft, bath, bhk)
    st.header(ans)
