import numpy as np
from scipy.spatial import cKDTree
import os.path
import sklearn.preprocessing

import utils
import preprocess

from keras.optimizers import SGD, Adam, RMSprop,Adadelta
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.utils import np_utils
from keras.models import model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

def make_model(inputs,hidden, outputs):

    input_img = Input(shape=(inputs,))
    encoded = Dense(2, activation='sigmoid')(input_img)
    encoded = Dense(2, activation='sigmoid')(encoded)

    decoded = Dense(2, activation='sigmoid')(encoded)
    decoded = Dense(inputs, activation='sigmoid')(decoded)

    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)
    #decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    autoencoder.summary()    
    # let's train the model using SGD + momentum (how original).
    opt = Adadelta() #lr=0.1)
    autoencoder.compile(loss='mse',optimizer=opt) #,              metrics=['accuracy'])
              
    return autoencoder,encoder
    


if __name__ == "__main__":
    alldata  = np.loadtxt("../data/muestras.csv",delimiter=",",skiprows=1)
    
    locations = alldata[:,0:3]
    data = alldata[:,3:5]
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    
    locations = scaler.fit_transform(locations)
    
    print locations.shape
    print data.shape
    
    n,nin = locations.shape
    n,nout = data.shape
    
    ret = utils.generate_kfold(range(n),n_folds=10,shuffle=True,random_state=1634120)    
    
    train_index, test_index = ret[0]
    
    print len(train_index),len(test_index)
    
    n_training = len(train_index)
    n_testing = len(test_index)

    model,encoder = make_model(nin,[100,100] ,nout)

    X_training = locations[train_index,:]
    y_training = data[train_index,:]
    X_testing = locations[test_index,:]
    y_testing = data[test_index,:]
        
    batch_size,nb_epoch = 5,1000

    history = model.fit(X_training, X_training,
        batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=1, validation_data=(X_testing, X_testing))
                        
    score = encoder.predict(X_testing)

    #print np.histogram(y_testing[:,0])
    #print np.histogram(score[:,0])
    #print np.histogram(y_testing[:,1])
    for o,p in zip(X_testing,score):
        print o,p
