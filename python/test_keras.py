import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os.path
import sklearn.preprocessing

import utils
import preprocess

from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.models import model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

def make_model_2(img_channels, img_rows, img_cols):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512,init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1,init='lecun_uniform'))
    model.add(Activation('linear'))

    model.summary()    
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.005) #, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse',
              optimizer=sgd,
              metrics=['accuracy'])
              
    return model

if __name__ == "__main__":
    alldata  = np.loadtxt("../data/gold.csv",delimiter=",",skiprows=1)
    alldata = alldata[:1000,:]
    
    locations = alldata[:,:2]
    data_original = alldata[:,2:]
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    
    data = scaler.fit_transform(data_original)
    
    if len(data.shape) < 2:
        data = np.expand_dims(data, axis=1)
    
    
    print data.shape
    
    n,m = data.shape
    
    ret = utils.generate_kfold(range(n),n_folds=5,shuffle=True,random_state=1634120)    
    
    train_index, test_index = ret[0]
    
    print len(train_index),len(test_index)
    
    n_training = len(train_index)
    n_testing = len(test_index)

    model = make_model_2(1, 101, 101)

    X_training = None
    y_training = None
    X_testing = None
    y_testing = None

    if os.path.exists("X_training.npy"):
        X_training = np.load("X_training.npy")
    if os.path.exists("y_training.npy"):
        y_training = np.load("y_training.npy")

    if os.path.exists("X_testing.npy"):
        X_testing = np.load("X_testing.npy")
    if os.path.exists("y_testing.npy"):
        y_testing = np.load("y_testing.npy")

    if X_training is None:
        X_training = np.empty((n_training,1,101,101))
        y_training = np.empty((n_training,1))
        X_testing = np.empty((n_testing,1,101,101))
        y_testing = np.empty((n_testing,1))
        
        kdtree = cKDTree(locations)
        k = 1000
        
        for i,index in enumerate(train_index):
            location = locations[index]
            image = preprocess.create_image_from_neighbours(location,locations,data,k,kdtree,(50,50),(60,72),distance=np.inf)
            X_training[i,:,:,:] = image
            y_training[i,:] = data[index,:]
        
        for i,index in enumerate(test_index):
            location = locations[index]
            image = preprocess.create_image_from_neighbours(location,locations,data,k,kdtree,(50,50),(60,72),distance=np.inf)
            X_testing[i,:,:,:] = image
            y_testing[i,:] = data[index,:]
            
        np.save("X_training",X_training)
        np.save("y_training",y_training)
        np.save("X_testing",X_testing)
        np.save("y_testing",y_testing)
        
    batch_size,nb_epoch = 100,10

    history = model.fit(X_training, y_training,
        batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=1, validation_data=(X_testing, y_testing))
                        
    score = model.evaluate(X_testing, y_testing, verbose=1)
