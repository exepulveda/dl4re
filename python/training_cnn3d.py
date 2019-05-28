import numpy as np
from scipy.spatial import cKDTree
import os.path
import sklearn.preprocessing
import sklearn.metrics
import sys

import utils
import preprocess

from keras.optimizers import SGD, Adam, RMSprop,Adagrad,Adadelta
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout3D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from keras.layers import Conv3D, MaxPooling3D,BatchNormalization

def make_model(img_channels, img_depth, img_rows, img_cols, neurons_full_layer,droprate=0.25):
    model = Sequential()
    data_format='channels_first'
    #print img_channels, img_rows, img_cols,neurons_full_layer

    model.add(Conv3D(4,(5,5,5),strides=(1,1,1), padding='valid',data_format=data_format,input_shape=(img_channels,img_depth, img_rows, img_cols) ))
    #model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(Conv3D(8,(3,3,3),strides=(1,1,1), padding='valid',data_format=data_format))
    #model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2),data_format=data_format))
    model.add(SpatialDropout3D(droprate,data_format=data_format))

    #model.add(Conv3D(8,(3,3,3),strides=(1,1,1), padding='valid',data_format=data_format))
    #model.add(BatchNormalization())    
    #model.add(Activation('relu'))
    #model.add(MaxPooling3D(pool_size=(2,2,2),data_format=data_format))
    #model.add(SpatialDropout3D(droprate,data_format=data_format))
        
    #model.add(Convolution2D(32, 5, 5,subsample=(2,2), border_mode='same',activation="relu"))
    #model.add(AveragePooling2D(pool_size=(2,2)))

    #model.add(Convolution2D(32, 2, 2, border_mode='valid',activation="relu"))
    #model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    #model.add(Dense(50,init='he_normal'))
    #model.add(Activation('relu'))

    model.add(Dense(100,kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    model.add(Dense(1,kernel_initializer='glorot_normal'))
    model.add(Activation('linear'))

    model.summary()    
    # let's train the model using SGD + momentum (how original).
    #opt = RMSprop(decay=0.9) #Adadelta() #Adagrad()
    #opt = SGD(lr=0.001, momentum=0.0, decay=0.9, nesterov=True)
    opt = Adam(lr=0.001)#Adagrad(0.001)
    model.compile(loss='mse',optimizer=opt) #,              metrics=['accuracy'])
              
    return model
    


def train(train_index, test_index,locations,data,wsize,wlength,epoch = 50,batch_size = 50,neurons_full_layer = 100,reuse=False):

    return r2_training,r2_testing
    

if __name__ == "__main__":
    n_args = len(sys.argv)
    
    epoch = 200
    batch_size = 64
    neurons_full_layer = 50
    
    if n_args > 1:
        epoch = int(sys.argv[1])
        
    if n_args > 2:
        batch_size = int(sys.argv[2])
        
    if n_args > 3:
        neurons_full_layer = int(sys.argv[3])
        
    #Loading data
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

    scaler_data = sklearn.preprocessing.StandardScaler()
    data_train = scaler_data.fit_transform(data_train)
    data_test = scaler_data.transform(data_test)
    
    all_locations = np.r_[locations_train,locations_test]
    all_data = np.r_[data_train,data_test]

    n,m = all_data.shape

    # CAMABIAR SEGUN CV
    wsize = np.array((10,10,10),dtype=np.int32)
    wlength = np.array((5.0,5.0,2.5))
    
    n_training = len(data_train)
    n_testing = len(data_test)

    nwsize = wsize*2+1
    
    reuse = False #True if you want to load the model and train it more epochs
    
    neurons_full_layer = 100
    
    if not reuse:
        model = make_model(m, nwsize[0], nwsize[1], nwsize[2],neurons_full_layer)
    else:
        model = utils.load_model("muestras-model-cnn3d-1")
        opt = Adam(lr=0.001)#Adagrad(0.001)
        model.compile(loss='mse',optimizer=opt)

    X_training = None
    y_training = None
    X_testing = None
    y_testing = None

    tag = "_cnn3d"

    if os.path.exists("X_training{}.npy".format(tag)):
        X_training = np.load("X_training{}.npy".format(tag))
    if os.path.exists("y_training{}.npy".format(tag)):
        y_training = np.load("y_training{}.npy".format(tag))

    if os.path.exists("X_testing{}.npy".format(tag)):
        X_testing = np.load("X_testing{}.npy".format(tag))
    if os.path.exists("y_testing{}.npy".format(tag)):
        y_testing = np.load("y_testing{}.npy".format(tag))
    

    if X_training is None:
        X_training = np.empty((n_training,m,nwsize[0],nwsize[1],nwsize[2]))
        y_training = np.empty((n_training,1))
        X_testing = np.empty((n_testing,m,nwsize[0],nwsize[1],nwsize[2]))
        y_testing = np.empty((n_testing,1))
        
        kdtree = cKDTree(all_locations)
        r = 100.0 #serach range
        
        print("processing training data")
        for i,location in enumerate(locations_train):
            image = preprocess.create_image_from_neighbours_3d(location,all_locations,all_data,kdtree,wsize,wlength,r)
            X_training[i,:,:,:] = image
            y_training[i,:] = data_train[i,:]
        
        print("processing testing data")
        for i,location in enumerate(locations_test):
            image = preprocess.create_image_from_neighbours_3d(location,all_locations,all_data,kdtree,wsize,wlength,r)
            X_testing[i,:,:,:] = image
            y_testing[i,:] = data_test[i,:]
            
        np.save("X_training{}".format(tag),X_training)
        np.save("y_training{}".format(tag),y_training)
        np.save("X_testing{}".format(tag),X_testing)
        np.save("y_testing{}".format(tag),y_testing)
    else:
        pass
        
        
    print(X_training.shape)
    print(X_testing.shape)
    print(y_training.shape)
    print(y_testing.shape)
    
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=0, mode='min', baseline=None, restore_best_weights=True)

    history = model.fit(X_training, y_training,batch_size=batch_size, epochs=epoch,verbose=1,validation_data=(X_testing, y_testing),callbacks=[es])
    
    utils.save_model(model,"muestras-model-cnn3d-1")

    #R2 in training
    prediction = model.predict(X_training)
    r2_training = sklearn.metrics.r2_score(y_training,prediction)

    #R2 in testing
    prediction = model.predict(X_testing)
    r2_testing = sklearn.metrics.r2_score(y_testing,prediction)

    print("r2:",r2_training,r2_testing)
    
