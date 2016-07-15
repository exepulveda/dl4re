import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os.path
import sklearn.preprocessing

import utils
import preprocess
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_3d(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')  


def make_model(x_image,y_value,channels, s1,s2,s3,features1,features2,pool_size,fullsize):
    #first Conv
    
    W_conv1 = weight_variable([5, 5, 5, channels, features1])
    b_conv1 = bias_variable([features1])

    h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_3d(h_conv1)

    #second Conv
    W_conv2 = weight_variable([5, 5, 5,features1, features2])
    b_conv2 = bias_variable([features2])

    h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_3d(h_conv2)
    
    shape_pool2 = h_pool2.get_shape()
    print "shape_pool2",shape_pool2
    
    fullconnected_size = 13 * 13 * 13 
    

    #fully connected layer
    fs1 = s1/2/2
    fs2 = s2/2/2
    fs3 = s3/2/2
    
    W_fc1 = weight_variable([fullconnected_size * features2, fullsize])
    b_fc1 = bias_variable([fullsize])

    h_pool2_flat = tf.reshape(h_pool2, [-1, fullconnected_size*features2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



    #linear final layer
    W_fc2 = weight_variable([fullsize, 1])
    b_fc2 = bias_variable([1])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2              


    #loss function
    loss = tf.nn.l2_loss(y_value - y_conv)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    return y_conv,loss,train_step,keep_prob
    


if __name__ == "__main__":
    alldata  = np.loadtxt("../data/muestras.csv",delimiter=",",skiprows=1)
    
    locations = alldata[:,0:3]
    data_original = alldata[:,3:4]
    
    #scaler = sklearn.preprocessing.MinMaxScaler()
    
    data = data_original #scaler.fit_transform(data_original)
    
    if len(data.shape) < 2:
        data = np.expand_dims(data, axis=1)
    
    
    kdtree = cKDTree(locations)


    nodes = (25,25,25)
    sizes = (10.0,10.0,10.0)
    
    print locations.shape
    print data.shape
    
    n,m = data.shape
    
    ret = utils.generate_kfold(range(n),n_folds=5,shuffle=True,random_state=1634120)    
    
    train_index, test_index = ret[0]
    
    print len(train_index),len(test_index)
    
    n_training = len(train_index)
    n_testing = len(test_index)


    print "loading test data..."
    X_testing = np.empty((n_testing,nodes[0]*2+1,nodes[1]*2+1,nodes[2]*2+1,1))
    y_testing = np.empty((n_testing,1))

    for k in range(n_testing):
        location = locations[test_index[k]]
        #values = data[train_index[k]]
        X_testing[k,:,:,:,:] = preprocess.create_image_from_neighbours_ts(location,locations,data,500,kdtree,nodes,sizes)
        y_testing[k,:] = data[test_index[k],:]

    print "building the model..."
    sess = tf.InteractiveSession()    
    
    x_image = tf.placeholder(tf.float32, [None, nodes[0]*2+1,nodes[1]*2+1,nodes[2]*2+1,1])
    y_value = tf.placeholder(tf.float32, [None, 1])

    y_conv,loss,train_step,keep_prob = make_model(x_image,y_value, 1, nodes[0]*2+1,nodes[1]*2+1,nodes[2]*2+1,32,64,None,50)
    
    sess.run(tf.initialize_all_variables())    
    
    batch_size = 10
    for i in range(100):
        
        steps = n_training // batch_size
        for j in range(0,n_training,batch_size):
            batch_starts = j
            batch_ends = min(n_training,batch_starts + batch_size)

            X_training = np.empty((batch_ends - batch_starts + 1,nodes[0]*2+1,nodes[1]*2+1,nodes[2]*2+1,1))
            y_training = np.empty((batch_ends - batch_starts + 1,1))

            
            print "training",batch_starts,batch_ends
            
            #make inputs
            print "making inputs..."
            for k in range(batch_starts,batch_ends):
                location = locations[train_index[k]]
                X_training[k,:,:,:,:] = preprocess.create_image_from_neighbours_ts(location,locations,data,500,kdtree,nodes,sizes)
                y_training[k,:] = data[train_index[k],:]
        
            print "one more step ..."
            train_step.run(feed_dict={x_image: X_training, y_value: y_training, keep_prob: 0.5})
            
            rms = loss.eval(feed_dict={x_image: X_testing, y_value: y_testing, keep_prob: 1.0})
            print "rms",rms


    #history = model.fit(X_training, y_training,
    #    batch_size=batch_size, nb_epoch=nb_epoch,
    #    verbose=1, validation_data=(X_testing, y_testing))
                        
    #score = model.predict(X_testing)
    
    #print np.mean(y_testing[:,0])
    #print np.mean(score[:,0])
    #print np.std(y_testing[:,0])
    #print np.std(score[:,0])
