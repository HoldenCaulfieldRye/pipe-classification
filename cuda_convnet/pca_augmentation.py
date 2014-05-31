import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import random
import time
from joblib import Parallel
from joblib import delayed

name = 'test.JPG'
size = (256,256)
channels = 3
N_JOBS = -1

''' Image here is a single row numpy array, with all of the channels in sequential
order (i.e. goes column by column then row by row for red, then same for green
then same for blue).
'''
def augment_image(image, random_gaussian):
    eigenvalue, eigenvector = np.linalg.eig(np.cov(image))
    addition = np.dot(eigenvector,np.sqrt(eigenvalue).T * random_gaussian)
    return np.clip(np.add(image.T,addition),0,255).reshape(-1)


''' Data here is a 2D matrix, with columns being images, and rows being pixels
of those images, in order of columns, then rows, starting with R, the G and
B in sequential order.  A matrix of similar shape is returned with the data
augmented to have different illumination levels via a PCA analysis & manipulation
'''
def augment_illumination(data,channels=3):
    data = data.reshape(-1,channels,data.shape[1]).T
    random_gaussian = np.random.normal(0,0.6,channels)
    rows = Parallel(n_jobs=N_JOBS)(
		    delayed(augment_image)(image, random_gaussian) 
		    for image in data)
    return np.vstack([r for r in rows]).T

def get_array(image,size):
    im = Image.open(image)
    im = ImageOps.fit(im, size, Image.ANTIALIAS)
    im_data = np.array(im)
    im_data = im_data.T.reshape(3, -1).reshape(-1)
    im_data = im_data.astype(np.single)
    return im_data



# Setup test data
data = get_array(name,size)
for i in range(0,7):
    data = np.vstack((data,data))
data=data.T
# Time data augmentation
start = time.time()
new_data = augment_illumination(data)
print 'Time taken:%.02f'%(time.time()-start)
