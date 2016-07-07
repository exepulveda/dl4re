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

def make_model(inputs,hidden, outputs):
    model = Sequential()

    model.add(Dense(hidden[0],init='lecun_uniform',input_shape=(inputs,),activation="sigmoid"))
    model.add(Dropout(0.2))

    for n in hidden[1:]:
        model.add(Dense(n,init='lecun_uniform',activation="sigmoid"))
        model.add(Dropout(0.2))

    model.add(Dense(outputs,init='lecun_uniform',activation="linear"))

    model.summary()    
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.001) #, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse',optimizer=sgd) #,              metrics=['accuracy'])
              
    return model
    


if __name__ == "__main__":
    alldata  = np.loadtxt("../data/muestras.csv",delimiter=",",skiprows=1)
    
    locations = alldata[:,0:2]
    data_original = alldata[:,3:5]
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    
    data = data_original #scaler.fit_transform(data_original)
    
    print locations.shape
    print data.shape
    
    n,nin = locations.shape
    n,nout = data.shape
    
    ret = utils.generate_kfold(range(n),n_folds=5,shuffle=True,random_state=1634120)    
    
    train_index, test_index = ret[0]
    
    print len(train_index),len(test_index)
    
    n_training = len(train_index)
    n_testing = len(test_index)

    model = make_model(nin,[100,100] ,nout)

    X_training = locations[train_index,:]
    y_training = data[train_index,:]
    X_testing = locations[test_index,:]
    y_testing = data[test_index,:]
        
    batch_size,nb_epoch = 10,100

    history = model.fit(X_training, y_training,
        batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=1, validation_data=(X_testing, y_testing))
                        
    score = model.predict(X_testing)
    
    print np.histogram(y_testing[:,0])
    print np.histogram(score[:,0])
    print np.histogram(y_testing[:,1])
    print np.histogram(score[:,1])
