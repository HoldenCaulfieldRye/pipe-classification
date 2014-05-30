import numpy as np
import os
import cPickle as pickle
from joblib import Parallel
from joblib import delayed

def get_a_batch_data_array():
    ''' loads an array of labels in the format expected by cuda-
    convnet, from a hard-coded location which on graphic02 corresponds
    to a suitable data file. (batches.meta?) '''

def get_a_pipe_data_list(data_dir):
    ''' loads the 10002.data file provided by ControlPoint and stores
    its contents in a list, to be returned. '''
    os.chdir(os.cwd()+data_dir)

def get_all_pipe_labels(data_dir):
    ''' looks into all .dat files in data_dir, and if find a new label
    , add it to the list. stores final list as binary pickle file.'''
    path = data_dir
    d = {'labels': []}
    for filename in os.listdir(path):
        if not filename.endswith('.dat'): continue
        with open('100002.dat') as f:
            content = f.readlines()
            for label in content:
                if label not in d['labels']:
                    d['labels'].append(label)
    d['no_labels'] = len(d['labels'])
    pickle.dump(d, open('labels.pickle', 'wb'))

def generate_xml_labels_from_pipe_data(data_dir):
    ''' creates .xml's from all .dat files in data_dir. '''
    path = os.getcwd()+data_dir
    for filename in os.listdir(path):
        if not filename.endswith('.dat'): continue
        # fullname = os.path.join(path, filename)
        dat = np.array(open(filename,'r'))

def generate_batches_from_pipe_data(data_dir, label_options):
    ''' generates data batches and batches.meta files in the format 
    expected by cuda-convnet, from a data format provided by 
    ControlPoint, from the location given by data_dir. label_options 
    indicates which labels to create: simple good/bad, or one for each
    characteristic. '''

    # 1) what should batch size be? needs to be optimal given:
    #    a) krizhevsky's gpu code
    #    b) size of the image (256x256, 224x224, or more?)
    #    c) data augmentation (if any) cpu time
    # 2) what should image size be? 
    # 3) does batches.meta contain the labels? or do these need to appear 
    #    inside each batch?

    # for now, keep label simple: good or bad weld

    # call a modified version of _collect_filenames_and_labels() from dataset.py, one that searches for .data files (and doesn't throw error if no xml's found)

    # alternatively! convert .data files to xml in same format as for plant, and let john's scripts do the hard work.



if __name__ == "__main__":
    import sys
    if sys.argv[1] == 'get_all_pipe_labels':
        get_all_pipe_labels(sys.argv[2])
