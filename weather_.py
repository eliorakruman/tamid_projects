import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import random

df = pd.read_csv("boston_weather_data.csv")
# print(df.head())  # shows the structure of the csv: time, tavg, tmin, tmax, prcp, wdir, wspd, pres. 3653 entries (3654 rows bc header)

''' p redicted precipitation amount in millimetres for next day '''

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

scaler = MinMaxScaler(feature_range=(0, 1))
rain_vals = scaler.fit_transform(df['prcp'].values.reshape(-1, 1))

sequence_length = 30  # past 14 days to predict the next day's weather

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

rain_vals = df['prcp'].values
X, y = create_sequences(rain_vals, sequence_length)

# 80% train, 20% test. don't shuffle bc weather is based on time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=SEED)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential([LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)), LSTM(50, activation='relu'), Dense(1)])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

last_days = rain_vals[-sequence_length:].reshape(1, sequence_length, 1)

predicted_rain = model.predict(last_days)

predicted_rain = scaler.inverse_transform(predicted_rain.reshape(-1, 1))
print(f"Predicted Precipitation: {predicted_rain[0][0]:.2f}")

print(f"Predicted Precipitation: {predicted_rain[0][0]:.2f}")

''' root mean squared error and R^2 metrics instead of accuracy '''

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")