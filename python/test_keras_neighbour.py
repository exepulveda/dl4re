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

def save_model(model,model_filename):
    print("saving model",model_filename)
    json_string = model.to_json()
    open(model_filename + ".json", 'w').write(json_string)
    model.save_weights(model_filename + ".h5",overwrite=True)

def load_model(model_filename):
    model = model_from_json(open(model_filename + ".json").read())    
    model.load_weights(model_filename + ".h5")

    return model

def make_model_locations(inputs,nh=[100],dropout_rate=0.3,activation='relu',lr=0.0001):
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
    opt = Adam(lr)
    model.compile(loss='mse',optimizer=opt) #,              metrics=['accuracy'])
              
    return model
    


def train(train_index, test_index,data,y_index,k=50,nh=[1000],epoch = 300,batch_size = 100):
    n,m = data.shape
    n_training = len(train_index)
    n_testing = len(test_index)

    kdtree = cKDTree(locations)

    model = make_model_locations(k*(3+m),nh=nh)

    X_training= preprocess.get_neighbours(locations[train_index,:],locations,data,k,kdtree,distance=np.inf)
    y_training = data[train_index,y_index]
    
    X_testing = preprocess.get_neighbours(locations[test_index,:],locations,data,k,kdtree,distance=np.inf)
    y_testing = data[test_index,y_index]
    
    #reshape
    X_training = X_training.reshape((n_training,k*(3+m)))
    X_testing = X_testing.reshape((n_testing,k*(3+m)))
    
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
    
            
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

    return r2_training,r2_testing
    

if __name__ == "__main__":
    alldata_original  = np.loadtxt("../data/muestras.csv",delimiter=",",skiprows=1)
    
    locations_original = alldata_original[:,0:3]
    data_original = alldata_original[:,3:4]

    if len(data_original.shape) < 2:
        data_original = np.expand_dims(data_original, axis=1)
    
    scale_data = True
    
    if scale_data:
        #scaler_locations = sklearn.preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
        #scaler_data = sklearn.preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
        scaler_locations = sklearn.preprocessing.StandardScaler()
        scaler_data = sklearn.preprocessing.StandardScaler()

        locations = scaler_locations.fit_transform(locations_original)
        data = scaler_data.fit_transform(data_original)
    else:
        locations = locations_original
        data = data_original

    
    n,m = data.shape

    k = 30
    
    #print locations.shape
    #print data.shape

    if True:
        folds = utils.generate_kfold(np.arange(n),n_folds=5,shuffle=True,random_state=1634120)    
        r2_folds = []
        for train_index, test_index in folds:
            r2_training,r2_testing = train(train_index, test_index,data,0,k=k,nh=[200,100,50],epoch = 500,batch_size = 32)
            print(r2_training,r2_testing)
            r2_folds += [r2_testing]
            
        print("r2 all folds:",r2_folds,np.mean(r2_folds))
    
