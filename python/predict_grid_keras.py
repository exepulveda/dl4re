import numpy as np
from scipy.spatial import cKDTree
import os.path
import sys

import utils
import preprocess

from keras.models import model_from_json


def load_model(model_filename):
    model = model_from_json(open(model_filename+".json").read())    
    model.load_weights(model_filename+".h5")

    return model


if __name__ == "__main__":
    starts = np.array([5.0,5.0,5.0])
    sizes = np.array([10.0,10.0,10.0])
    nodes = np.array([40,60,13],dtype=np.int)
    nx,ny,nz = nodes
    nxy = nx*ny
    nxyz = nxy*nz
    
    alldata  = np.loadtxt("../data/muestras.csv",delimiter=",",skiprows=1)
    
    locations = alldata[:,0:3]
    data = alldata[:,3:4]
    kdtree = cKDTree(locations)

    wsize = np.array((20,20,20),dtype=np.int32)
    nwsize = wsize*2+1

    kn = 1000

    predictions = np.empty((nx,ny,nz))
    model = load_model("muestras-model")

    for k in range(nodes[2]):
        #plant
        X_training = np.empty((nxy,1,nwsize[0],nwsize[1],nwsize[2]))
        
        
        print "processing predicting data",k
        for i in xrange(nx):
            for j in xrange(ny):
                location = [sizes[0]*i + starts[1],sizes[1]*j + starts[2],sizes[2]*k + starts[2]]
                image = preprocess.create_image_from_neighbours_3d(location,locations,data,kn,kdtree,(wsize[0],wsize[1],wsize[2]),(10.0,10.0,10.0),distance=np.inf)
                X_training[i,:,:,:,:] = image
        
        print "predicting",k
        score = model.predict(X_training)
        
        print score
    
        predictions[:,:,k] = score
    
    np.savetxt("predictions.csv",predictions,fmt="%0.4",delimiter=",")
