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

    model.add(Convolution2D(16, 4, 4, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    
    #model.add(Convolution2D(64, 3, 3, border_mode='same'))
    #model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(100,init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(img_channels,init='lecun_uniform'))
    model.add(Activation('linear'))

    model.summary()    
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.001) #, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse',optimizer=sgd) #,              metrics=['accuracy'])
              
    return model
    


if __name__ == "__main__":
    alldata  = np.loadtxt("../data/muestras.csv",delimiter=",",skiprows=1)
    
    locations = alldata[:,0:2]
    data_original = alldata[:,3:4]
    
    #scaler = sklearn.preprocessing.MinMaxScaler()
    
    data = data_original #scaler.fit_transform(data_original)
    
    if len(data.shape) < 2:
        data = np.expand_dims(data, axis=1)
    
    
    print locations.shape
    print data.shape
    
    n,m = data.shape
    
    ret = utils.generate_kfold(range(n),n_folds=5,shuffle=True,random_state=1634120)    
    
    train_index, test_index = ret[0]
    
    print len(train_index),len(test_index)
    
    n_training = len(train_index)
    n_testing = len(test_index)

    model = make_model_2(m, 101, 101)

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
        X_training = np.empty((n_training,m,101,101))
        y_training = np.empty((n_training,m))
        X_testing = np.empty((n_testing,m,101,101))
        y_testing = np.empty((n_testing,m))
        
        kdtree = cKDTree(locations)
        k = 1000
        
        print "processing training data"
        for i,index in enumerate(train_index):
            location = locations[index]
            image = preprocess.create_image_from_neighbours(location,locations,data,k,kdtree,(50,50),(5,5),distance=np.inf)
            X_training[i,:,:,:] = image
            y_training[i,:] = data[index,:]
        
        print "processing testing data"
        for i,index in enumerate(test_index):
            location = locations[index]
            image = preprocess.create_image_from_neighbours(location,locations,data,k,kdtree,(50,50),(5,5),distance=np.inf)
            X_testing[i,:,:,:] = image
            y_testing[i,:] = data[index,:]
            
        np.save("X_training",X_training)
        np.save("y_training",y_training)
        np.save("X_testing",X_testing)
        np.save("y_testing",y_testing)
    else:
        
    batch_size,nb_epoch = 100,50

    history = model.fit(X_training, y_training,
        batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=1, validation_data=(X_testing, y_testing))
                        
    score = model.predict(X_testing)
    
    print np.mean(y_testing[:,0])
    print np.mean(score[:,0])
    print np.std(y_testing[:,0])
    print np.std(score[:,0])
