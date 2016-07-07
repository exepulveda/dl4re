import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def create_image_from_neighbours(location,locations,data,k,kdtree,nodes,sizes,distance=np.inf):
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
    indices = np.where(n_image>0)
    image[indices] = values_image[indices] / n_image[indices]

    return image

if __name__ == "__main__":
    alldata  = np.loadtxt("../data/gold.csv",delimiter=",",skiprows=1)
    
    locations = alldata[:,:2]
    data = alldata[:,2:]
    
    if len(data.shape) < 2:
        data = np.expand_dims(data, axis=1)
    
    
    print data.shape
    
    n,m = (100,1)
    
    #data = np.random.random((n,m))
    #location = [626030.6165,604192.74395]
    location = [640676.233,621889.5938]
    kdtree = cKDTree(locations)
    k = 1000
    image = create_image_from_neighbours(location,locations,data,k,kdtree,(50,50),(60,72),distance=np.inf)
    
    plt.imshow(image[0,:,:])
    plt.show()
