import os, sys, shutil
import cPickle as pickle
import numpy as np
from PIL import Image

if __name__ == '__main__':
    # get directory from which to fetch data
    try: 
        get_dir = sys.argv[3]
    except: get_dir = 'test_data/example_ensemble/Two/'

    # get filename from which to unpickle image(s)
    try: data_filename = sys.argv[2]
    except: data_filename = 'data_batch_1'

    # get pickled objects
    os.chdir(get_dir)
    pickle_label = open('batches.meta')
    pickle_imgs = open(data_filename)

    # unpickle
    meta = pickle.load(pickle_label)
    batch = pickle.load(pickle_imgs)

    # potentially do stuff to the mean
    try:
        if sys.argv[1] == '+mean': batch['data'] += meta['data_mean']
        elif sys.argv[1] == '-mean': batch['data'] -= meta['data_mean']
        else:
            print 'Error: 3rd command line argument not recognised.'
            print 'assuming default behaviour: not doing anything to the mean.'
    except: pass

    # unflatten
    print 'batch[\'data\'] needs to have shape', batch['data'].shape
    batch['data'] = batch['data'].reshape(batch['data'].shape[1], 256, 256, 3) # shape[1]?
    batch['data'] = np.require(batch['data'], dtype=np.uint8, requirements='W')

    # get first image, save it, to know what original looks like
    orig_img_np = batch['data'][0].copy() # is this image demeaned?
    orig_img_jpg = Image.fromarray(orig_img_np)
    os.chdir('../')
    try:
        os.mkdir('Alex')
        os.chdir('Alex/')
    except:
        os.chdir('Alex/')
        os.remove('orig_img.jpeg')
    orig_img_jpg.save('orig_img.jpeg') # saving jpg
    print 'the image the constitutes the batch was stored as orig_img.jpg in:\n', os.getcwd()
    print ''

    # pickle 1 image batch, save it
    batch['data'] = batch['data'][0].copy()
    batch['labels'] = batch['labels'][0].copy()
    pickle.dump(batch['data'], open('data_batch_1', 'wb'))
    pickle.dump(batch['labels'], open('batches.meta', 'wb'))
    print 'cuda-convnet compatible data stored in:\n', os.getcwd()
    print 'you can pass this directory to plantdataprovider to test it'


