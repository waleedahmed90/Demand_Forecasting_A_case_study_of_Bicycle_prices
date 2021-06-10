import pandas as pd
import sys
from utils import *
import joblib
import warnings
warnings.filterwarnings('ignore')

# Read existing data
existing_data = pd.read_csv('dataset/ml_training.csv')
training_len = len(existing_data)

# Read new data
new_file = sys.argv[1]
test_data = pd.read_csv(new_file)
forecast_len = len(test_data)

# Load the ML model
model_filename = sys.argv[2]

# Combine the training and forecast datatset
all_data = existing_data.append(test_data, ignore_index=True)


# preprocess - rename, categorize, change_index, remove_corr
df = ml_preprocess(all_data)


# Split the dataset
X_train, Y_train, X_test, Y_test = ml_train_test_split(df, training_len, forecast_len)

# Encode data
X_train = ml_encode_dataset(X_train)
X_test = ml_encode_dataset(X_test)

# Fit the model on the training data (We fit it everytime new data is added)
for col in ['month', 'weekday']:
    X_train[col] = X_train[col].astype('int64')

model = joblib.load(model_filename)
temp = model.fit(X_train, Y_train, verbose=False)
joblib.dump(model, model_filename)


# Model forecast - output values in a new file
for col in ['month', 'weekday']:
    X_test[col] = X_test[col].astype('int64')
pred = model.predict(X_test)


print(pred)
