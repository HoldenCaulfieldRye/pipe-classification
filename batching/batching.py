import numpy as n
import os

def get_a_batch_data_array():
    ''' loads an array of labels in the format expected by cuda-convnet, 
    from a hard-coded location which on graphic02 corresponds to a 
    suitable data file. (batches.meta?) '''

def get_a_pipe_data_list(data_dir):
    ''' loads the 10002.data file provided by ControlPoint and stores
    its contents in a list, to be returned. '''
    os.chdir(os.cwd()+data_dir)
    
def generate_batches_from_pipe_data(data_dir, label_options):
    ''' generates data batches and batches.meta files in the format 
    expected by cuda-convnet, from a data format provided by ControlPoint, 
    from the location given by data_dir. label_options indicates which
    labels to create: simple good/bad, or one for each characteristic. '''

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
