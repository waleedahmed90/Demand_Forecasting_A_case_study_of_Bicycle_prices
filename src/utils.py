import pandas as pd
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
import numpy as np
tqdm().pandas()

def ml_preprocess(dataset):
    # Rename column names
    dataset.rename(columns={'instant':'rec_id','dteday':'datetime','yr':'year','mnth':'month','weathersit':'weather_condition',
    'hum':'humidity','cnt':'total_count'}, inplace=True)
    dataset.drop('rec_id', axis=1, inplace=True) # Drop record column

    # Categorize dataset
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset['datetime'] = pd.to_datetime(dataset.datetime)
    dataset['season'] = dataset.season.astype('category')
    dataset['year'] = dataset.year.astype('category')
    dataset['month'] = dataset.month.astype('category')
    dataset['holiday'] = dataset.holiday.astype('category')
    dataset['weekday'] = dataset.weekday.astype('category')
    dataset['workingday'] = dataset.workingday.astype('category')
    dataset['weather_condition'] = dataset.weather_condition.astype('category')
    dataset.set_index('datetime', inplace=True)

    # Remove highly correlated features
    dataset.drop(['casual','registered','temp'], axis = 1, inplace = True)
    dataset.reset_index(inplace=True)
    return dataset

def ml_encode_dataset(dataset):
    #Create a new dataset for test attributes
    all_attributes = dataset[['season','month','year','weekday','holiday','workingday','humidity','atemp','windspeed','weather_condition']]
    #categorical attributes
    cat_attributes = ['season','holiday','workingday','weather_condition','year']
    #numerical attributes
    num_attributes = ['atemp','windspeed','humidity','month','weekday']

    encoded_attributes = pd.get_dummies(all_attributes,columns=cat_attributes)
    return encoded_attributes

def ml_train_test_split(dataset, n_split_train, n_split_forecast):
    # Split the dataset into train and test in 70:30 ratio
    train_set = dataset[:n_split_train]
    test_set = dataset[-n_split_forecast:]

    X_train, Y_train = train_set.iloc[:,0:-1], train_set.iloc[:,-1]
    X_test, Y_test = test_set.iloc[:,0:-1], test_set.iloc[:,-1]

    return X_train, Y_train, X_test, Y_test




def nn_preprocess(dataset):
    # Rename column names
    dataset.rename(columns={'instant':'rec_id','dteday':'datetime','yr':'year','mnth':'month','weathersit':'weather_condition',
    'hum':'humidity','cnt':'total_count'}, inplace=True)
    dataset.drop('rec_id', axis=1, inplace=True) # Drop record column

    # Categorize dataset
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset['datetime'] = pd.to_datetime(dataset.datetime)
    dataset['season'] = dataset.season.astype('category')
    dataset['year'] = dataset.year.astype('category')
    dataset['hr'] = dataset.hr.astype('category')
    dataset['month'] = dataset.month.astype('category')
    dataset['holiday'] = dataset.holiday.astype('category')
    dataset['weekday'] = dataset.weekday.astype('category')
    dataset['workingday'] = dataset.workingday.astype('category')
    dataset['weather_condition'] = dataset.weather_condition.astype('category')

    # Remove highly correlated features
    dataset.drop(['casual','registered','temp', 'datetime'], axis = 1, inplace = True)
    dataset.reset_index(inplace=True)
    return dataset

def nn_encode_dataset(dataset):
    #Create a new dataset for train attributes
    all_attributes = dataset[['season','month','hr', 'year','weekday','holiday','workingday','weather_condition','humidity','atemp','windspeed', 'total_count']]

    #categorical attributes
    cat_attributes=['season','holiday','hr','workingday','weather_condition','year']

    #numerical attributes
    num_attributes=['atemp','windspeed','humidity','month','weekday']

    data = pd.get_dummies(all_attributes,columns=cat_attributes)
    return data

def nn_train_test_split(dataset):
    train, test = train_test_split(dataset, test_size=0.1, random_state=0)
    return train, test

def prepare_data(X,y,time_steps=1):
    Xs = []
    Ys = []
    for i in tqdm(range(len(X) - time_steps)):
        a = X.iloc[i:(i + time_steps)].to_numpy()
        Xs.append(a)
        Ys.append(y.iloc[i+time_steps])
    return np.array(Xs),np.array(Ys)
