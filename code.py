import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import multiprocessing
from joblib import Parallel, delayed
import time
import matplotlib.pyplot as plt


start_time = time.time()


def prepare_data(df, sequence_length=60):
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    training_data_len = int(len(data) * 0.8)
    train_data = scaled_data[0:training_data_len, :]

    x_train, y_train = [], []
    for i in range(sequence_length, len(train_data)):
        x_train.append(train_data[i-sequence_length:i, 0])
        y_train.append(train_data[i, 0])

    return np.array(x_train), np.array(y_train), scaler

# creting the model
def create_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=False),
        Dense(16),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# to parallel process
def process_chunk(chunk, sequence_length):
    x, y = [], []
    for i in range(sequence_length, len(chunk)):
        x.append(chunk[i-sequence_length:i, 0])
        y.append(chunk[i, 0])
    return np.array(x), np.array(y)

def parallel_data_preparation(data, sequence_length, n_jobs=-1):
    chunks = np.array_split(data, multiprocessing.cpu_count())
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk, sequence_length) for chunk in chunks
    )
    x = np.concatenate([r[0] for r in results])
    y = np.concatenate([r[1] for r in results])
    return x, y


if __name__ == "__main__":
    # importing the goole stock data here
    df = pd.read_csv('/content/drive/MyDrive/GOOG_5yr.csv') 

    
    x_train, y_train, scaler = prepare_data(df)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = create_model((x_train.shape[1], 1))

    # multi_GPU
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model((x_train.shape[1], 1))

    # early stoping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # training the model
    history = model.fit(
        x_train, y_train,
        batch_size=32 * strategy.num_replicas_in_sync,  
        epochs=200,
        validation_split=0.2,
        callbacks=[early_stopping],
        use_multiprocessing=True,
        workers=multiprocessing.cpu_count()
    )

    
    test_data = df.filter(['Close']).values[-60:]  
    test_data = scaler.transform(test_data)
    x_test = np.array([test_data])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)

    print(f"Predicted price for upcoming day : {prediction[0][0]} ")

    end_time = time.time()
    print(f"Time taken to run: {end_time - start_time} seconds")



#actual_prices = df['Close'].values

#actual_prices = scaler.inverse_transform(x_train)
#predicted_prices = prediction

data = df.filter(['Close']).values
training_data_len = int(len(data) * 0.8)
y_test = data[training_data_len:, :]


scaled_data = scaler.fit_transform(data)
test_data = scaled_data[training_data_len - 10:, :]
x_test = []
for i in range(10, len(test_data)):
    x_test.append(test_data[i-10:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


plt.figure(figsize=(12, 6))
plt.plot(y_test, label='actual close')
plt.plot(predictions, label='predicted close')
plt.title('actual vs predicted using GOOGLE past 5yr data')
plt.xlabel('days')
plt.ylabel('Price')
plt.legend()
plt.show()