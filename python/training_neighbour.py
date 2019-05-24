import numpy as np
from scipy.spatial import cKDTree
import os.path
import sklearn.preprocessing
import sklearn.metrics
import sys

sys.path += ["../../geostatpy"]
import geometry

import utils
import preprocess

from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.models import model_from_json
from keras.callbacks import EarlyStopping

def make_model_locations(inputs,nh=[100],dropout_rate=0.5,activation='relu',lr=0.001):
    model = Sequential()

    model.add(Dense(nh[0],kernel_initializer='glorot_normal',input_shape=(inputs,)))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    for n in nh[1:]:
        model.add(Dense(n,kernel_initializer='glorot_normal'))
        model.add(Activation(activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1,kernel_initializer='glorot_normal'))
    model.add(Activation('linear'))

    model.summary()    
    # let's train the model using SGD + momentum (how original).
    opt = Adam(lr) #Adam(lr)
    model.compile(loss='mse',optimizer=opt) #,              metrics=['accuracy'])
              
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

    locations_train = scaler_locations.fit_transform(locations_train)
    data_train = scaler_data.fit_transform(data_train)

    locations_test = scaler_locations.transform(locations_test)
    data_test = scaler_data.transform(data_test)
    
    all_locations = np.r_[locations_train,locations_test]
    all_data = np.r_[data_train,data_test]

    
    n,m = data_train.shape

    k = 30
    nh =[200,100,50]
    dropout = 0.5
    activation = 'relu'
    batch_size = 16
    epoch = 1000
    lr = 0.0001

    n_training = len(data_train)
    n_testing = len(data_test)

    kdtree = cKDTree(all_locations)

    model = make_model_locations(k*(3+m),nh=nh,dropout_rate=dropout,activation=activation,lr=lr)

    X_training= preprocess.get_neighbours(locations_train,all_locations,all_data,k,kdtree,distance=np.inf)
    y_training = data_train
    
    X_testing = preprocess.get_neighbours(locations_test,all_locations,all_data,k,kdtree,distance=np.inf)
    y_testing = data_test
    
    #reshape
    X_training = X_training.reshape((n_training,k*(3+m)))
    X_testing = X_testing.reshape((n_testing,k*(3+m)))
    
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    
            
    history = model.fit(X_training, y_training,
        batch_size=batch_size, epochs=epoch,
        verbose=1, validation_data=(X_testing, y_testing),shuffle=True,callbacks=[es])

    utils.save_model(model,"muestras-model-neighbour")

    #R2 in training
    prediction = model.predict(X_training)
    r2_training = sklearn.metrics.r2_score(y_training,prediction)

    #R2 in testing
    prediction = model.predict(X_testing)
    r2_testing = sklearn.metrics.r2_score(y_testing,prediction)

    print(r2_training,r2_testing)
