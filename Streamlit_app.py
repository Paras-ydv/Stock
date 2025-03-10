import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import setuptools #used to have distutils feature bcz it was removed from python after 3.12 
import pandas_datareader as data
import yfinance as yf 
import streamlit as st
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.models import load_model

start = '2010-01-01'
end = '2019-12-31'

#Sets the title of the Streamlit app.
st.title("Stock Trend Prediction")

#Taking the stock name as input from the app user
user_input = st.text_input("Enter Stock Ticker: ".upper(), 'AAPL')
#Downloading the historical data of stock from yahoo finance
df = yf.download(user_input,start,end)

#Describing the data to the user
st.subheader("Data from 2010-2019")
st.write(df.describe())

#Visualizations
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.xlabel("Time Chart")
plt.ylabel("Closing Price")
st.pyplot(fig)


st.subheader("Closing Price vs Time Chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0: int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) :int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Load the model
model = load_model('keras_model.h5')

#Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

#Final Graph
st.subheader("Predictions vs Original")
plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label = "Original Price")
plt.plot(y_predicted, 'r', label = "Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig)