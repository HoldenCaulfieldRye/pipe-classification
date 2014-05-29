import os, sys, shutil
import cPickle as pickle
import numpy as np
from PIL import Image, ImageOps

# sys.path.append('../../noccn/noccn/')
# import datasetNoMongo as dataset

data_dir = 'Alex/'

if __name__ == '__main__':
    # get image location
    try: directory = sys.argv[1]
    except: directory = 'test_data/'
    try: img_filename = sys.argv[2]
    except: img_filename = '11.jpg'

    # get numpy array of image
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    img_jpg = Image.open(directory+img_filename).convert("RGB")
    img_jpg = ImageOps.fit(img_jpg, (256, 256), Image.ANTIALIAS)
    img_np = np.array(img_jpg)
    img_np = img_np.T.reshape(3, -1).reshape(-1)
    img_np = img_np.astype(np.single)
    print 'data is in shape:', img_np.shape

    # get label, metadata (hacky)
    start = os.getcwd()
    os.chdir('test_data/example_ensemble/Two/')
    meta = pickle.load(open('batches.meta'))
    batch = pickle.load(open('data_batch_1'))
    os.chdir(start)
    # make data_mean zero because we don't want to demean 1-img data
    meta['data_mean'] = np.zeros(img_np.shape, dtype=np.float32)
    batch['labels'] = np.array([[1]]) # too many brackets?
    batch['data'] = np.vstack([img_np]).T # that's how dataset.py does it
    # make sure dimensions ok
    print 'just made a batch. data shape is %s, labels shape is %s' % (batch['data'].shape, batch['labels'].shape)
    
    os.chdir(data_dir)

    # pickle dat shit
    pickle.dump(batch, open('data_batch_1', 'wb'))
    pickle.dump(meta, open('batches.meta', 'wb'))
    print '1-img batch stored in:\n', os.getcwd()
    print 'you can pass this directory to plantdataprovider to test it'
    os.chdir('../../../')


