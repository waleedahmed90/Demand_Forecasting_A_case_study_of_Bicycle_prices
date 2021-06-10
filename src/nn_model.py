import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

from utils import *
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout , LSTM , Bidirectional
import sys
import scipy.stats as st

# Read existing data
existing_data = pd.read_csv('dataset/nn_training.csv')
training_len = len(existing_data)

# Read new data
new_file = sys.argv[1]
test_data = pd.read_csv(new_file)
forecast_len = len(test_data)

# Load the ML model
model_filename = sys.argv[2]
model_weights = sys.argv[3]

# Combine the training and forecast datatset
all_data = existing_data.append(test_data, ignore_index=True)


# preprocess - rename, categorize, change_index, remove_corr
df = nn_preprocess(all_data)

# Encode dataset
df_encoded = nn_encode_dataset(df)


# Split the datatset
train, test = nn_train_test_split(df_encoded)


# Normalize the data to make it compatible for a Neural Network(NN)
scaler  = MinMaxScaler()

num_colu = ['atemp', 'humidity', 'windspeed']
trans_1 = scaler.fit(train[num_colu].to_numpy())
train.loc[:,num_colu] = trans_1.transform(train[num_colu].to_numpy())
test.loc[:,num_colu] = trans_1.transform(test[num_colu].to_numpy())

cnt_scaler = MinMaxScaler()
trans_2 = cnt_scaler.fit(train[["total_count"]])
train["total_count"] = trans_2.transform(train[["total_count"]])
test["total_count"] = trans_2.transform(test[["total_count"]])

# Scale the data to make it compatible for a NN
steps=24
X_train , Y_train = prepare_data(train, train.total_count, time_steps=steps)
X_test , Y_test = prepare_data(test, test.total_count, time_steps=steps)

# Fit the model

# Load the saved model
json_file = open(model_filename, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_weights)

model.compile(optimizer="adam",loss="mse")
prepared_model = model.fit(X_train,Y_train,batch_size=32,epochs=100,validation_data=(X_test,Y_test))

# Save the latest trained model
model_json = model.to_json()
with open(model_filename, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_weights)


# Model forecast - output values in a new file
pred = model.predict(X_test)
pred_inv = cnt_scaler.inverse_transform(pred)


print(pred_inv.reshape(-1))
