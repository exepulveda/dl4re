import numpy as np
from scipy.spatial import cKDTree
import os.path
import sklearn.preprocessing
import sklearn.metrics
import sys

sys.path += ["/home/esepulveda/projects/geostatpy"]
import geometry


import utils
import preprocess

from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.models import model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization

def save_model(model,model_filename):
    print("saving model",model_filename)
    json_string = model.to_json()
    open(model_filename + ".json", 'w').write(json_string)
    model.save_weights(model_filename + ".h5",overwrite=True)

def load_model(model_filename):
    model = model_from_json(open(model_filename + ".json").read())    
    model.load_weights(model_filename + ".h5")

    return model

def make_model_locations(nh=100):
    model = Sequential()

    model.add(Dense(nh,init='lecun_uniform',input_shape=(3,)))
    model.add(Activation('sigmoid'))

    model.add(Dense(1,init='lecun_uniform'))
    model.add(Activation('linear'))

    model.summary()    
    # let's train the model using SGD + momentum (how original).
    opt = Adagrad()
    model.compile(loss='mse',optimizer=opt) #,              metrics=['accuracy'])
              
    return model
    

def train():
    print sys.argv
    n_args = len(sys.argv)
    
    nb_epoch = 300
    batch_size = 100
    
    if n_args > 1:
        nb_epoch = int(sys.argv[1])
        
    if n_args > 2:
        batch_size = int(sys.argv[2])
        
    
    
    alldata_original  = np.loadtxt("../data/muestras.csv",delimiter=",",skiprows=1)
    scaler_data = sklearn.preprocessing.MinMaxScaler(feature_range=(0.1, 0.9))
    alldata = scaler_data.fit_transform(alldata_original)    
    
    locations = alldata[:,0:3]
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

    model = make_model_locations(nh=1000)

    X_training = np.empty((n_training,3))
    y_training = np.empty((n_training,m))
    X_testing = np.empty((n_testing,3))
    y_testing = np.empty((n_testing,m))
    
    X_training[:,:] = locations[train_index,:]
    y_training[:,:] = data[train_index,:]
      
    X_testing[:,:] = locations[test_index,:]
    y_testing[:,:] = data[test_index,:]
            

    history = model.fit(X_training, y_training,
        batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=1, validation_data=(X_testing, y_testing),shuffle=True)

    save_model(model,"muestras-model-locations")

    score = model.predict(X_testing)
    
    print np.mean(y_testing[:,0])
    print np.mean(score[:,0])
    print np.std(y_testing[:,0])
    print np.std(score[:,0])

    print "r2", sklearn.metrics.r2_score(score, y_testing)

    quit()
    print "Testing values"
    for true_value, prediction in zip(y_testing[:,0],score[:,0]):
        print true_value,',', prediction

    print "Training values"
    score = model.predict(X_training)
    for true_value, prediction in zip(y_training[:,0],score[:,0]):
        print true_value,',', prediction

def predict():
    model.predict(X_testing)

if __name__ == "__main__":
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
    
    nodes = [40,60,1]
    sizes = [10.0,10.0,10.0]
    starts = [5.0,5.0,125.0]
    
    grid = geometry.Grid3D(nodes,sizes,starts)
    
    dgrid = grid.discretize([4,4,1])
    
    nbatch = len(dgrid)

    locations_batch = np.empty((nbatch,3))
    
    model = load_model("muestras-model-locations")
    
    predictions = np.empty(len(grid))

    for i,p in enumerate(grid):
        locations_batch = scaler_locations.transform(dgrid + p)
        
        p = model.predict(locations_batch)
        
        p = scaler_data.inverse_transform(p)
        
        predictions[i] = np.mean(p)
        
        print predictions[i]

    #print "r2", sklearn.metrics.r2_score(score, y_testing)

