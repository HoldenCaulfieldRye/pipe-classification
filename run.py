import cPickle as pickle
from fnmatch import fnmatch
import operator
import os
import random
import traceback
import math
import csv
from multiprocessing import Process
from itertools import tee, izip_longest
import sys
import time
import numpy as np
from PIL import Image
from PIL import ImageOps
from joblib import Parallel
from joblib import delayed
# This import path ensures the appropriate modules are available
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "cuda_convnet"))
import convnet
import options
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "noccn/noccn"))
from script import *
# This is used to parse xml files
import xml.etree.ElementTree as ET # can be speeded up using lxml possibly
import xml.dom.minidom as minidom


# Exit errors to be returned via sys.exit when the
# program does not successfully complete all jobs
NO_ERROR = 0
COULD_NOT_OPEN_IMAGE_FILE = 1
COULD_NOT_START_CONVNET = 2
COULD_NOT_SAVE_OUTPUT_FILE = 3
INVALID_COMMAND_ARGS = 4
SERVER_ERROR = 5

class MyError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


# Accepts an image filename, and number of channels,
# processes the image into a 1D numpy array, of the
# form [R G B] with each colour collapsed into row major order
def _process_tag_item(size,channels,name):
    try:
        im = Image.open(name)
        im = ImageOps.fit(im, size, Image.ANTIALIAS)
        im_data = np.array(im)
        im_data = im_data.T.reshape(channels, -1).reshape(-1)
        im_data = im_data.astype(np.single)
        return im_data
    except:
        raise MyError(COULD_NOT_OPEN_IMAGE_FILE)


# Yields chunks of a specified size n of a list until it
# is empty.  Chunks are not guaranteed to be of size n
# if the list is not a multiple of the chunk size
def chunks(l, n):
    for i in xrange(0, len(l), n):
            yield l[i:i+n]


# Returns the next item in a list, at the same time as
# the first item.  If there is no next, it returns None
def get_next(some_iterable):
    it1, it2 = tee(iter(some_iterable))
    next(it2)
    return izip_longest(it1, it2)


# Class which runs through for a given batch of a single type
# the network defined in the run.cfg file.
class ImageRecogniser(object):
    def __init__(self,batch_size=128,channels=3,size=(256,256),
                  model=None,n_jobs=-1,**kwargs):
        self.batch_size = batch_size
        self.channels = channels
        self.size = size
        self.n_jobs = n_jobs
        self.model = model
        vars(self).update(**kwargs) 

    # Main processing function. It works from the list of filenames
    # passed in, in 128 chunks, processing into numpy arrays and 
    # classifying with the classifier
    def __call__(self, filenames):
        batch_num = 1
        batch_means = np.zeros(((self.size[0]**2)*self.channels,1))
        start_time = time.clock()
        for filenames,next_filenames in get_next(list(chunks(filenames,self.batch_size))):
            if batch_num == 1:
                rows = Parallel(n_jobs=self.n_jobs)(
                       delayed(_process_tag_item)(self.size,self.channels,filename)
                       for filename in filenames)
            data = np.vstack([r for r in rows if r is not None]).T
            if data.shape[0] < len(filenames):    
                raise MyError(COULD_NOT_OPEN_IMAGE_FILE)
            if data.shape[1] == 1:
                mean = np.mean(data)
            elif data.shape[1] > 1:
                mean = data.mean(axis=1).reshape(((self.size[0]**2)*self.channels,1))
            data = data - mean
            if self.model is not None:
                self.model.start_predictions(data)
            if next_filenames is not None:
                rows = Parallel(n_jobs=self.n_jobs)(
                    delayed(_process_tag_item)(self.size,self.channels,filename)
                    for filename in next_filenames)
            try:    
                if self.model is not None:
                    self.model.finish_predictions(filenames)
                else:
                    pass
            except:
                raise MyError(COULD_NOT_SAVE_OUTPUT_FILE)
            batch_num += 1
        return NO_ERROR    
        
        

# The wrapper class for the convnet which has already been
# trained.  Which convnet gets loaded is determined by the
# run.cfg file.  It will finish a batch by pickle dumping
# each of the image files results to a *.pickle equivilent
# to the *.jpg that was given. The set size that will be in
# that result will vary between convulutional nets.  The 
# combine script takes care of reizing with appropriate spaces.
class PlantConvNet(convnet.ConvNet):
    def __init__(self, op, load_dic, dp_params={}):
        convnet.ConvNet.__init__(self,op,load_dic,dp_params)
        self.softmax_idx = self.get_layer_idx('probs', check_type='softmax')
        self.tag_names = list(self.test_data_provider.batch_meta['label_names'])
        self.b_data = None
        self.b_labels = None
        self.b_preds = None


    def import_model(self):
        self.libmodel = __import__("_ConvNet") 


    def start_predictions(self, data):
        # If multiview take patches
        if self.multiview_test:
            data_dim = 150528
            border_size = 16
            inner_size = 224
            num_views = 5*2
            target = np.zeros((data_dim,data.shape[1]*num_views),dtype=np.single)
            y = data.reshape(3, 256, 256, data.shape[1])
            start_positions = [(0,0),  (0, border_size*2), (border_size, border_size), (border_size*2, 0), (border_size*2, border_size*2)] 
            end_positions = [(sy+inner_size, sx+inner_size) for (sy,sx) in start_positions]
            for i in xrange(num_views/2): 
                pic = y[:,start_positions[i][0]:end_positions[i][0], 
                        start_positions[i][1]:end_positions[i][1],:]
                target[:,i * data.shape[1]:(i+1)* data.shape[1]] = pic.reshape((data_dim,data.shape[1])) 
                target[:,(num_views/2 + i) * data.shape[1]:(num_views/2 +i+1)* data.shape[1]] = pic[:,:,::-1,:].reshape((data_dim,data.shape[1])) 
            data = target    

        # Run the batch through the model
        self.b_data = np.require(data, requirements='C')
        self.b_labels = np.zeros((1, data.shape[1]), dtype=np.single)
        self.b_preds = np.zeros((data.shape[1], len(self.tag_names)), dtype=np.single)
        self.libmodel.startFeatureWriter([self.b_data, self.b_labels, self.b_preds], self.softmax_idx)


    def finish_predictions(self, filenames):
        # Finish the batch
        self.finish_batch()
        # Combine results for multiview test
        if self.multiview_test:
            num_views = 5*2
            num_images = self.b_labels.shape[1]/num_views
            processed_preds = np.zeros((num_images,len(self.tag_names)))
            for image in range(0,num_images):
                tmp_preds = self.b_preds[image::num_images]
                processed_preds[image] = tmp_preds.T.mean(axis=1).reshape(tmp_preds.T.shape[0],-1).T
            self.b_preds = processed_preds    
        for filename,row in zip(filenames,self.b_preds):
            file_storage = open(os.path.splitext(filename)[0] + '.pickle','wb')
            pickle.dump(np.array(row),file_storage)
            file_storage.close()


    @classmethod
    def get_options_parser(cls):
        op = convnet.ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('load_file'):
                op.delete_option(option)
        return op


def get_recogniser(cfg,component):
    cfg_options_file = cfg.get(component,'Type classification not found')
    cfg_data_options = get_options(cfg_options_file, 'dataset')
    try:
        conv_model = make_model(PlantConvNet,'run',cfg_options_file)
    except:
        raise MyError(COULD_NOT_START_CONVNET)
    run = ImageRecogniser(
        batch_size=int(cfg.get('batch-size', 128)),
        channels=int(cfg_data_options.get('channels', 3)),
        size=eval(cfg_data_options.get('size', '(256, 256)')),
        model=conv_model,
        threshold=float(cfg.get('threshold',0.0)),
        )
    return run

# The console interpreter.  It checks whether the arguments
# are valid, and also parses the configuration files.
def console(config_file = None):
    if len(sys.argv) < 3:
        print 'Must give a component type and valid image file as arguments'
        raise MyError(INVALID_COMMAND_ARGS)
    cfg = get_options(os.path.dirname(os.path.abspath(__file__))+'/run.cfg', 'run')
    valid_args = cfg.get('valid_args','entire,stem,branch,leaf,fruit,flower').split(',')
    if sys.argv[1] not in valid_args:
        print 'First argument must be one of: [',
        for arg in valid_args:
            print arg + ' ',
        print ']'
        raise MyError(INVALID_COMMAND_ARGS)
    run = get_recogniser(cfg,sys.argv[1])
    run(sys.argv[2:])


# Boilerplate for running the appropriate function.
if __name__ == "__main__":
    console()
