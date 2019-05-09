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

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation

from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.callbacks import EarlyStopping
#from keras.utils import np_utils
#from keras.models import model_from_json
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.layers.convolutional import Convolution3D, MaxPooling3D
#from keras.layers.normalization import BatchNormalization

def save_model(model,model_filename):
    print("saving model",model_filename)
    json_string = model.to_json()
    open(model_filename + ".json", 'w').write(json_string)
    model.save_weights(model_filename + ".h5",overwrite=True)

def load_model(model_filename):
    model = model_from_json(open(model_filename + ".json").read())    
    model.load_weights(model_filename + ".h5")

    return model

def make_model_locations(nh,dropout_rate=0.5,activation='relu',lr=0.0001):
    model = Sequential()
    
    model.add(Dense(nh[0],kernel_initializer='glorot_normal',input_dim=4))
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
    opt = Adam(lr)
    model.compile(loss='mse',optimizer=opt) #,              metrics=['accuracy'])
              
    return model
    

def train(train_index,test_index,locations,data,epoch = 300,batch_size = 100,nh=[1000],dropout_rate=0.5):
    n,m = data.shape
    print(n,m)
    
    n_training = len(train_index)
    n_testing = len(test_index)

    model = make_model_locations(nh=nh,dropout_rate=dropout_rate)

    X_training = np.empty((n_training,3+m-1))
    y_training = np.empty((n_training,1))
    X_testing = np.empty((n_testing,3+m-1))
    y_testing = np.empty((n_testing,1))
    
    X_training[:,:3] = locations[train_index,:]
    X_training[:,3] = data[train_index,1] #au
    y_training[:,0] = data[train_index,0] #cu
      
    X_testing[:,:3] = locations[test_index,:]
    X_testing[:,3] = data[test_index,1]
    y_testing[:,0] = data[test_index,0]
            

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

    history = model.fit(X_training, y_training,
        batch_size=batch_size, epoch=epoch,
        verbose=1, validation_data=(X_testing, y_testing),shuffle=True,callbacks=[es])

    utils.save_model(model,"muestras-model-locations")

    #R2 in training
    prediction = model.predict(X_training)
    r2_training = sklearn.metrics.r2_score(y_training,prediction)


    prediction = model.predict(X_testing)
    r2_testing = sklearn.metrics.r2_score(y_testing,prediction)

    return r2_training,r2_testing

if __name__ == "__main__":
    from keras import backend as K
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)))
    
    #Using all data and then spliting it
    alldata_original  = np.loadtxt("../data/muestras.csv",delimiter=",",skiprows=1)
    
    scaler_locations = sklearn.preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
    scaler_data = sklearn.preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
    
    locations_original = alldata_original[:,0:3]
    data_original = alldata_original[:,3:5]
    
    locations = scaler_locations.fit_transform(locations_original)
    data = scaler_data.fit_transform(data_original)
    
    if len(data.shape) < 2:
        data = np.expand_dims(data, axis=1)
    
    print(locations.shape)
    print(data.shape)

    n,m = data.shape

    if True:
        fold = utils.generate_kfold(np.arange(n),n_folds=5,shuffle=True,random_state=1634120)    
        r2_folds = []
        for train_index, test_index in fold:
            r2_training,r2_testing = train(train_index,test_index,locations,data,nh=[100,50],nb_epoch=10000,dropout_rate=0.3)
            print(r2_training,r2_testing)
            r2_folds += [r2_testing]
            
        print("r2 all folds:",np.mean(r2_folds))

    if not True:
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
            
            print(predictions[i])

        #print "r2", sklearn.metrics.r2_score(score, y_testing)

