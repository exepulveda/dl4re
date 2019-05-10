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
    n,m = data.shape
    
    print(len(train_index),len(test_index))
    
    n_training = len(train_index)
    n_testing = len(test_index)

    nwsize = wsize*2+1
    
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
        
        kdtree = cKDTree(locations)
        k = 1000
        
        print("processing training data")
        for i,index in enumerate(train_index):
            location = locations[index]
            image = preprocess.create_image_from_neighbours_3d(location,index,locations,data,kdtree,wsize,wlength,100.0)
            X_training[i,:,:,:] = image
            y_training[i,:] = data[index,0]
        
        print("processing testing data")
        for i,index in enumerate(test_index):
            location = locations[index]
            image = preprocess.create_image_from_neighbours_3d(location,index,locations,data,kdtree,wsize,wlength,100.0)
            X_testing[i,:,:,:] = image
            y_testing[i,:] = data[index,0]
            
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

    return r2_training,r2_testing
    
def test():
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
    
    print(X_training.shape)
    print(X_testing.shape)
    print(y_training.shape)
    print(y_testing.shape)

    model = utils.load_model("muestras-model-cnn3d-1")

    #score = model.predict(X_testing)
    
    #print(np.mean(y_testing[:,0]))
    #print(np.mean(score[:,0]))
    #print(np.std(y_testing[:,0]))
    #print(np.std(score[:,0]))

    #print("r2", sklearn.metrics.r2_score(y_testing,score))

    #score = model.predict(X_training)
    
    #print("r2", sklearn.metrics.r2_score(y_training,score))
    n_training = len(X_training)

    all_X = np.concatenate((X_training, X_testing), axis=0)
    all_y = np.concatenate((y_training, y_testing), axis=0)
    
    del X_training
    del X_testing

    score = model.predict(all_X)
    
    print(np.mean(all_y[:,0]))
    print(np.mean(score[:,0]))
    print(np.std(all_y[:,0]))
    print(np.std(score[:,0]))

    print("r2", sklearn.metrics.r2_score(all_y,score))    

    for i in range(len(score)):
        y_true = all_y[i,0]
        y_pred = score[i,0]
        print(y_true,y_pred,0 if i < n_training else 1,sep=',')

if __name__ == "__main__":
    from keras import backend as K
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)))
    
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
    
    alldata_original  = np.loadtxt("../data/muestras.csv",delimiter=",",skiprows=1)
    
        
    locations_original = alldata_original[:,0:3]
    data_original = alldata_original[:,3]
    if len(data_original.shape) < 2:
        data_original = np.expand_dims(data_original, axis=1)
        
    scale_data = True
    
    if scale_data:
        #scaler_locations = sklearn.preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
        #scaler_data = sklearn.preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
        #scaler_locations = sklearn.preprocessing.StandardScaler()
        scaler_data = sklearn.preprocessing.StandardScaler()

        #locations = scaler_locations.fit_transform(locations_original)
        data = scaler_data.fit_transform(data_original)
    else:
        data = data_original
    
    locations = locations_original

    if len(data.shape) < 2:
        data = np.expand_dims(data, axis=1)
    
    n,m = data.shape

    wsize = np.array((10,10,10),dtype=np.int32)
    wlength = np.array((5.0,5.0,2.5))
    

    
    fold = utils.generate_kfold(np.arange(n),n_folds=5,shuffle=True,random_state=1634120)    
    r2_folds = []
    for train_index, test_index in fold:
        r2_training,r2_testing = train(train_index, test_index,locations,data,wsize,wlength,epoch = epoch,batch_size = batch_size,neurons_full_layer = neurons_full_layer,reuse=False)    
        print(r2_training,r2_testing)
        r2_folds += [r2_testing]
        
    print("r2 all folds:",np.mean(r2_folds))
    
