import numpy as np
from scipy.spatial import cKDTree
import os.path
import sklearn.preprocessing
import sklearn.metrics
import sys

import utils
import preprocess

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation

from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.callbacks import EarlyStopping

def make_model_locations(nh,dropout_rate=0.5,activation='relu',lr=0.001):
    model = Sequential()
    
    model.add(Dense(nh[0],kernel_initializer='glorot_normal',input_dim=3))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    if len(nh)>1:
        for n in nh[1:]:
            model.add(Dense(units=n,kernel_initializer='glorot_normal'))
            model.add(Activation(activation))
            model.add(Dropout(dropout_rate))

    model.add(Dense(units=1,kernel_initializer='glorot_normal',activation='linear'))

    model.summary()    
    # let's train the model using SGD + momentum (how original).
    opt = Adam(lr,decay=0.001)
    
    model.compile(opt,loss="mse")
    
    return model

if __name__ == "__main__":
    #Using all data and then spliting it
    train_original  = np.loadtxt("../data/training_branco.csv",delimiter=";",skiprows=1)
    test_original  = np.loadtxt("../data/testing_branco.csv",delimiter=";",skiprows=1)
    
    
    locations_train = train_original[:,0:3]
    data_train      = train_original[:,3]

    locations_test = test_original[:,0:3]
    data_test      = test_original[:,3]

    print(np.mean(data_train))
    print(np.mean(data_test))

    
    if len(data_train.shape) < 2:
        data_train = np.expand_dims(data_train, axis=1)
        data_test = np.expand_dims(data_test, axis=1)


    scaler_locations = sklearn.preprocessing.StandardScaler()
    scaler_data = sklearn.preprocessing.StandardScaler()



    #scaling only 
    locations_train = scaler_locations.fit_transform(locations_train)
    data_train = scaler_data.fit_transform(data_train)

    locations_test = scaler_locations.transform(locations_test)
    data_test = scaler_data.transform(data_test)

    n,m = data_train.shape
    

    
    
    n_training = len(data_train)
    n_testing = len(data_test)
    
    nh = [500,300]
    dropout_rate = 0.3
    lr = 0.002
    activation = 'relu'
    epoch = 10000
    batch_size = 32

    model = make_model_locations(nh=nh,dropout_rate=dropout_rate,lr=lr,activation=activation)

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

    history = model.fit(locations_train, data_train,
        batch_size=batch_size, epochs=epoch,
        verbose=1, validation_data=(locations_test, data_test),shuffle=True,callbacks=[es])

    utils.save_model(model,"muestras-model-locations")

    #R2 in training
    prediction = model.predict(locations_train)
    r2_training = sklearn.metrics.r2_score(data_train,prediction)

    #R2 in testing
    prediction = model.predict(locations_test)
    r2_testing = sklearn.metrics.r2_score(data_test,prediction)
    
    print(r2_training,r2_testing)

