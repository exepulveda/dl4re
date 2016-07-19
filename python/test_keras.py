import numpy as np
from scipy.spatial import cKDTree
import os.path
import sklearn.preprocessing
import sklearn.metrics
import sys

import utils
import preprocess

from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution3D, MaxPooling3D

def make_model_2(img_channels, img_rows, img_cols,d3, neurons_full_layer):
    model = Sequential()

    print img_channels, img_rows, img_cols,d3

    model.add(Convolution3D(32, 4, 4,4, border_mode='same', input_shape=(img_channels, img_rows, img_cols,d3)))
    model.add(MaxPooling3D(pool_size=(2, 2,2)))
    model.add(Activation('relu'))
    
    #model.add(Convolution2D(64, 3, 3, border_mode='same'))
    #model.add(Activation('relu'))
    model.add(Convolution3D(64, 4, 4,4))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2,2)))

    #model.add(Convolution3D(32, 4, 4,4))
    #model.add(Activation('relu'))
    #model.add(MaxPooling3D(pool_size=(2, 2,2)))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(neurons_full_layer,init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(img_channels,init='lecun_uniform'))
    model.add(Activation('linear'))

    model.summary()    
    # let's train the model using SGD + momentum (how original).
    opt = Adam()
    model.compile(loss='mse',optimizer=opt) #,              metrics=['accuracy'])
              
    return model
    


def train(train_index, test_index,locations,data,wsize,nb_epoch = 50,batch_size = 50,neurons_full_layer = 100):
    n,m = data.shape
    
    print len(train_index),len(test_index)
    
    n_training = len(train_index)
    n_testing = len(test_index)

    nwsize = wsize*2+1

    model = make_model_2(m, nwsize[0], nwsize[1],nwsize[2],neurons_full_layer)

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
        X_training = np.empty((n_training,m,nwsize[0],nwsize[1],nwsize[2]))
        y_training = np.empty((n_training,m))
        X_testing = np.empty((n_testing,m,nwsize[0],nwsize[1],nwsize[2]))
        y_testing = np.empty((n_testing,m))
        
        kdtree = cKDTree(locations)
        k = 1000
        
        print "processing training data"
        for i,index in enumerate(train_index):
            location = locations[index]
            image = preprocess.create_image_from_neighbours_3d(location,locations,data,k,kdtree,(wsize[0],wsize[1],wsize[2]),(10.0,10.0,10.0),distance=np.inf)
            X_training[i,:,:,:,:] = image
            y_training[i,:] = data[index,:]
        
        print "processing testing data"
        for i,index in enumerate(test_index):
            location = locations[index]
            image = preprocess.create_image_from_neighbours_3d(location,locations,data,k,kdtree,(wsize[0],wsize[1],wsize[2]),(10.0,10.0,10.0),distance=np.inf)
            X_testing[i,:,:,:,:] = image
            y_testing[i,:] = data[index,:]
            
        np.save("X_training",X_training)
        np.save("y_training",y_training)
        np.save("X_testing",X_testing)
        np.save("y_testing",y_testing)
    else:
        pass
        

    history = model.fit(X_training, y_training,
        batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=1, validation_data=(X_testing, y_testing))
        
    utils.save_model(model,"muestras-model-cnn")        
                        
    score = model.predict(X_testing)
    
    print np.mean(y_testing[:,0])
    print np.mean(score[:,0])
    print np.std(y_testing[:,0])
    print np.std(score[:,0])

    print "r2", sklearn.metrics.r2_score(score, y_testing)

def predict(locations,data):
    print sys.argv
    n_args = len(sys.argv)
    
    nb_epoch = 1
    batch_size = 50
    neurons_full_layer = 100
    
    if n_args > 1:
        nb_epoch = int(sys.argv[1])
        
    if n_args > 2:
        batch_size = int(sys.argv[2])
        
    if n_args > 3:
        neurons_full_layer = int(sys.argv[3])
    
    
    
    alldata  = np.loadtxt("../data/muestras.csv",delimiter=",",skiprows=1)
    
    locations = alldata[:,0:3]
    data_original = alldata[:,3:4]
    
    #scaler = sklearn.preprocessing.MinMaxScaler()
    
    data = data_original #scaler.fit_transform(data_original)
    
    if len(data.shape) < 2:
        data = np.expand_dims(data, axis=1)
    
    
    print locations.shape
    print data.shape
    
    n,m = data.shape
    
    ret = utils.generate_kfold(range(n),n_folds=10,shuffle=True,random_state=1634120)    
    
    train_index, test_index = ret[0]
    
    print len(train_index),len(test_index)
    
    n_training = len(train_index)
    n_testing = len(test_index)

    wsize = np.array((20,20,20),dtype=np.int32)
    nwsize = wsize*2+1

    print wsize
    print nwsize

    model = make_model_2(m, nwsize[0], nwsize[1],nwsize[2],neurons_full_layer)

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
        X_training = np.empty((n_training,m,nwsize[0],nwsize[1],nwsize[2]))
        y_training = np.empty((n_training,m))
        X_testing = np.empty((n_testing,m,nwsize[0],nwsize[1],nwsize[2]))
        y_testing = np.empty((n_testing,m))
        
        kdtree = cKDTree(locations)
        k = 1000
        
        print "processing training data"
        for i,index in enumerate(train_index):
            location = locations[index]
            image = preprocess.create_image_from_neighbours_3d(location,locations,data,k,kdtree,(wsize[0],wsize[1],wsize[2]),(10.0,10.0,10.0),distance=np.inf)
            X_training[i,:,:,:,:] = image
            y_training[i,:] = data[index,:]
        
        print "processing testing data"
        for i,index in enumerate(test_index):
            location = locations[index]
            image = preprocess.create_image_from_neighbours_3d(location,locations,data,k,kdtree,(wsize[0],wsize[1],wsize[2]),(10.0,10.0,10.0),distance=np.inf)
            X_testing[i,:,:,:,:] = image
            y_testing[i,:] = data[index,:]
            
        np.save("X_training",X_training)
        np.save("y_training",y_training)
        np.save("X_testing",X_testing)
        np.save("y_testing",y_testing)
    else:
        pass
        

    history = model.fit(X_training, y_training,
        batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=1, validation_data=(X_testing, y_testing))
                        
    score = model.predict(X_testing)
    
    print np.mean(y_testing[:,0])
    print np.mean(score[:,0])
    print np.std(y_testing[:,0])
    print np.std(score[:,0])

    print "Testing values"
    for true_value, prediction in zip(y_testing[:,0],score[:,0]):
        print true_value,',', prediction

    print "Training values"
    score = model.predict(X_training)
    for true_value, prediction in zip(y_training[:,0],score[:,0]):
        print true_value,',', prediction


if __name__ == "__main__":
    print sys.argv
    n_args = len(sys.argv)
    
    nb_epoch = 2
    batch_size = 50
    neurons_full_layer = 100
    
    if n_args > 1:
        nb_epoch = int(sys.argv[1])
        
    if n_args > 2:
        batch_size = int(sys.argv[2])
        
    if n_args > 3:
        neurons_full_layer = int(sys.argv[3])
    
    alldata_original  = np.loadtxt("../data/muestras.csv",delimiter=",",skiprows=1)
    
    scaler_locations = sklearn.preprocessing.MinMaxScaler(feature_range=(0.1, 0.9))
    scaler_data = sklearn.preprocessing.MinMaxScaler(feature_range=(0.1, 0.9))
    
    locations_original = alldata_original[:,0:3]
    data_original = alldata_original[:,3:4]
    
    locations = scaler_locations.fit_transform(locations_original)
    data = scaler_data.fit_transform(data_original)

    if len(data.shape) < 2:
        data = np.expand_dims(data, axis=1)
    
    
    print locations.shape
    print data.shape
    
    n,m = data.shape
    
    ret = utils.generate_kfold(range(n),n_folds=10,shuffle=True,random_state=1634120)    
    
    train_index, test_index = ret[0]
    
    print len(train_index),len(test_index)
    
    n_training = len(train_index)
    n_testing = len(test_index)
    
    #train_index = train_index[:100]
    #test_index = test_index[:100]

    wsize = np.array((20,20,20),dtype=np.int32)
    
    train(train_index, test_index,locations,data,wsize,nb_epoch = nb_epoch,batch_size = batch_size,neurons_full_layer = neurons_full_layer)    
