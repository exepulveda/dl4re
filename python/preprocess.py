'''This module is a helper module with several functions for imputing spatial data as a 2d or 3d image
'''

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

"""This function assigns samples into a 2D grid"""
def create_image_from_neighbours_2d(location,locations,data,k,kdtree,nodes,sizes,distance=np.inf):
    #few validations
    assert len(nodes) == len(sizes)
    
    #print nodes,sizes
    
    #first create image using nodes to both directions
    
    image_shape_ori = np.array(nodes,dtype=np.int16) * 2 + 1
    
    grid_size = np.array(sizes)

    n,m = data.shape
    
    image_shape = [m] + list(image_shape_ori)
    image_shape_tmp = [m] + list(image_shape_ori)

    values_image = np.zeros(image_shape_tmp)
    n_image = np.zeros(image_shape_ori,dtype=np.int32)
    distances_image = np.zeros(image_shape_ori)
    image = np.zeros(image_shape)
    
    distances,indices = kdtree.query(location,k=k,distance_upper_bound=distance)
    
    max_distance = np.sqrt(np.sum((np.array(nodes) * np.array(sizes)))**2)
    
    for i in xrange(k):
        if np.isfinite(distances[i]) and distances[i] > 0:
            #valid
            index = indices[i]
            coord = locations[index]
            diff = (coord - location)
            #indices of diff
            grid_indices = np.int32(np.floor(diff/sizes))
            distance = np.sqrt(np.sum(diff**2))
            if grid_indices[0] > -nodes[0] and grid_indices[0] < nodes[0] and grid_indices[1] > -nodes[1] and grid_indices[1] < nodes[1]:
                #print index,grid_indices,data[index,:],values_image.shape
                #image[:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1]] = data[index,:]
                values_image[:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1]] += data[index,:]
                n_image[grid_indices[0]+nodes[0],grid_indices[1]+nodes[1]] += 1
                distances_image[grid_indices[0]+nodes[0],grid_indices[1]+nodes[1]] += (distance / max_distance)
                #print grid_indices,distance/max_distance,data[index,:]

    #print distances_image[np.where(n_image>0)]
    #plt.imshow(distances_image,cmap = plt.get_cmap('gray'))
    #plt.show()
    #quit()

                    
    #find n_images > 0
    indices = np.where(n_image>0)
    for j,k in zip(*indices):
        image[:,j,k] = values_image[:,j,k] / n_image[j,k]
        #image[-1,:,:] = distances_image[j,k] / n_image[j,k] / 520.0

    #image_show = np.moveaxis(image,0,-1)
   
    #print image_show.shape
    
    #plt.imshow(image_show[:,:,0])
    
    #plt.show()
    #quit()

    return image

"""This function assigns samples into a 3D grid"""
def create_image_from_neighbours_3d(location,index_loc,locations,cont_data,data_cat,kdtree,nodes,sizes,distance):
    #few validations
    assert len(nodes) == len(sizes)
    
    #first create an image centered at location with length size in all directions
    image_shape = np.array(nodes,dtype=np.int16) * 2 + 1

    grid_size = np.array(sizes)

    n,m = cont_data.shape
    n,ncat = data_cat.shape

    #image has 4 dimensions (features,nx,ny,nz)
    image_shape = [m+ncat] + list(image_shape)

    values_image = np.zeros(image_shape)
    n_image = np.zeros(image_shape,dtype=np.int32)
    image = np.zeros(image_shape)
    
    if kdtree is not None:
        #fill cells with neighbourhood
        indices = kdtree.query_ball_point(location,distance)
            
        for index in indices:
            if index != index_loc:
                #valid
                coord = locations[index]
                diff = (coord - location)
                #indices of diff
                grid_indices = np.int32(np.floor(diff/sizes))
                if (-nodes[0] < grid_indices[0] < nodes[0]) and (-nodes[1] < grid_indices[1] < nodes[1]) and (-nodes[2] < grid_indices[2] < nodes[2]):
                    #print index,grid_indices,data[index,:]
                    #image[:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1]] = data[index,:]
                    values_image[:m,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1],grid_indices[2]+nodes[2]] = cont_data[index,:]
                    values_image[m:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1],grid_indices[2]+nodes[2]] = data_cat[index,:]
                    #n_image[:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1],grid_indices[2]+nodes[2]] += 1
                    
    #find n_images > 0
    #indices = np.where(n_image>0)
    #report mean value
    #image[indices] = values_image[indices] / n_image[indices]

    return values_image
    
"""This function assigns samples into a batch of 3D grid"""
def create_image_from_neighbours_3d_batch(location_batch,locations,cont_data,data_cat,k,kdtree,nodes,sizes,distance=np.inf):
    #few validations
    assert len(nodes) == len(sizes)
    
    #first create image using nodes to both directions
    
    image_shape = np.array(nodes,dtype=np.int16) * 2 + 1

    grid_size = np.array(sizes)

    n,m = cont_data.shape
    n,ncat = data_cat.shape

    image_shape = [m+ncat] + list(image_shape)

    values_image = np.zeros(image_shape)
    n_image = np.zeros(image_shape,dtype=np.int32)
    image = np.zeros(image_shape)
    
    #making a kdtree of location_batch
    location_batch_kdtree = cKDTree(location_batch)
    
    if kdtree is not None:
        distances,indices = kdtree.query(location_batch_kdtree,k=k,distance_upper_bound=distance)
        
        #print distances
            
        for i in xrange(k):
            if np.isfinite(distances[i]) and distances[i] > 0:
                #valid
                index = indices[i]
                coord = locations[index]
                diff = (coord - location)
                #indices of diff
                grid_indices = np.int32(np.floor(diff/sizes))
                if grid_indices[0] > -nodes[0] and grid_indices[0] < nodes[0] and grid_indices[1] > -nodes[1] and grid_indices[1] < nodes[1] and grid_indices[2] > -nodes[2] and grid_indices[2] < nodes[2]:
                    #print index,grid_indices,data[index,:]
                    #image[:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1]] = data[index,:]
                    values_image[:m,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1],grid_indices[2]+nodes[2]] = cont_data[index,:]
                    values_image[m:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1],grid_indices[2]+nodes[2]] = data_cat[index,:]
                    #n_image[:m,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1],grid_indices[2]+nodes[2]] += 1
                    
    #find n_images > 0
    #indices = np.where(n_image>0)
    #image[indices] = values_image[indices] / n_image[indices]
    
    return values_image

def create_image_from_neighbours_3d_batch(location_batch,locations,data,k,kdtree,nodes,sizes,distance=np.inf):
    #few validations
    assert len(nodes) == len(sizes)
    
    #first create image using nodes to both directions
    
    image_shape = np.array(nodes,dtype=np.int16) * 2 + 1

    grid_size = np.array(sizes)

    n,m = data.shape

    image_shape = [m] + list(image_shape)

    values_image = np.zeros(image_shape)
    n_image = np.zeros(image_shape,dtype=np.int32)
    image = np.zeros(image_shape)
    
    #making a kdtree of location_batch
    location_batch_kdtree = cKDTree(location_batch)
    
    if kdtree is not None:
        distances,indices = kdtree.query(location_batch_kdtree,k=k,distance_upper_bound=distance)
        
        #print distances
            
        for i in xrange(k):
            if np.isfinite(distances[i]) and distances[i] > 0:
                #valid
                index = indices[i]
                coord = locations[index]
                diff = (coord - location)
                #indices of diff
                grid_indices = np.int32(np.floor(diff/sizes))
                if grid_indices[0] > -nodes[0] and grid_indices[0] < nodes[0] and grid_indices[1] > -nodes[1] and grid_indices[1] < nodes[1] and grid_indices[2] > -nodes[2] and grid_indices[2] < nodes[2]:
                    #print index,grid_indices,data[index,:]
                    #image[:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1]] = data[index,:]
                    values_image[:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1],grid_indices[2]+nodes[2]] += data[index,:]
                    n_image[:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1],grid_indices[2]+nodes[2]] += 1
                    
    #find n_images > 0
    indices = np.where(n_image>0)
    image[indices] = values_image[indices] / n_image[indices]

    return image

def create_sparse_image_from_neighbours(location,locations,data,k,kdtree,nodes,sizes,distance=np.inf):
    #few validations
    assert len(nodes) == len(sizes)
    
    #first create image using nodes to both directions
    
    image_shape = np.array(nodes,dtype=np.int16) * 2 + 1

    grid_size = np.array(sizes)

    n,m = data.shape

    image_shape = [m] + list(image_shape)

    values_image = np.zeros(image_shape)
    n_image = np.zeros(image_shape,dtype=np.int32)
    
    if kdtree is not None:
        distances,indices = kdtree.query(location,k=k,distance_upper_bound=distance)
        
        #print distances
            
        for i in xrange(k):
            if np.isfinite(distances[i]) and distances[i] > 0:
                #valid
                index = indices[i]
                coord = locations[index]
                diff = (coord - location)
                #indices of diff
                grid_indices = np.int32(np.floor(diff/sizes))
                if grid_indices[0] > -nodes[0] and grid_indices[0] < nodes[0] and grid_indices[1] > -nodes[1] and grid_indices[1] < nodes[1]:
                    #print index,grid_indices,data[index,:]
                    #image[:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1]] = data[index,:]
                    values_image[:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1]] += data[index,:]
                    n_image[:,grid_indices[0]+nodes[0],grid_indices[1]+nodes[1]] += 1
                    
    #find n_images > 0
    sparse_image = []
    indices = np.where(n_image>0)
    for index in indices:
        sparse_image += [(index,values_image[index] / n_image[index])]

    return sparse_image

def get_neighbours(location_batch,locations,data,k,kdtree,distance=np.inf):
    #few validations
    
    #first create image using nodes to both directions
    
    nbatch,location_size = location_batch.shape
    n,m = data.shape 
    '''m is the number of features, therefore each neigbour has
    location size (2 or 3) + m features
    '''

    
    neighbourhood = np.zeros((nbatch,k,1+m))

    distances,indices = kdtree.query(location_batch,k=k+1,distance_upper_bound=distance)
    
    #print distances.shape,indices.shape #(n,k)

    #print(distances[0],indices[0])
    #quit()
    
    for i in range(nbatch):
        #print neighbourhood[i,:,0:location_size].shape,locations[indices[i,:],:].shape
        idx = indices[i,1:]
        neighbourhood[i,:,0] = np.linalg.norm(location_batch[i] - locations[idx,:],axis=1)
        neighbourhood[i,:,1:] = data[idx]

    #neighbourhood[:,:,indices] = 0.0

    return neighbourhood


if __name__ == "__main__":
    #alldata  = np.loadtxt("../data/gold.csv",delimiter=",",skiprows=1)
    alldata  = np.loadtxt("../data/muestras.csv",delimiter=",",skiprows=1)
    
    locations = alldata[:,:3]
    data = alldata[:,3:4]
    
    print(data.shape)
    
    k = 10
    kdtree = cKDTree(locations)
    
    location_batch = np.empty((2,3))
    location_batch[0,:] = [50,50,50]
    location_batch[1,:] = [0,0,0]
    
    neighbourhood = get_neighbours(location_batch,locations,data,k,kdtree,distance=np.inf)
    print(neighbourhood)
    
    k = 10
    kdtree = cKDTree(locations)
    
    location_batch = np.empty((2,3))
    location_batch[0,:] = [50,50,50]
    location_batch[1,:] = [0,0,0]
    
    neighbourhood = get_neighbours(location_batch,locations,data,k,kdtree,distance=np.inf)
    
    n,m = (100,1)
    
    #data = np.random.random((n,m))
    #location = [626030.6165,604192.74395]
    #location = [640676.233,621889.5938]
    #kdtree = cKDTree(locations)
    #k = 1000
    #image = create_image_from_neighbours(location,locations,data,k,kdtree,(50,50),(60,72),distance=np.inf)
    
    #plt.imshow(image[0,:,:])
    #plt.show()

