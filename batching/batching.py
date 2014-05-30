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
    whichBox = data_dir.split('/')[-1]
    d = {'labels': []}
    for filename in os.listdir(path):
        if not filename.endswith('.dat'): continue
        fullname = os.path.join(path, filename)
        with open(fullname) as f:
            content = f.readlines()
            for label in content:
                if label not in d['labels']:
                    print label
                    d['labels'].append(label)
    d['labels'].sort()
    d['no_labels'] = len(d['labels'])
    pickle.dump(d, open('labels'+whichBox+'.pickle', 'wb'))
    print 'saved pickle file in', os.getcwd()

def generate_xml_labels_from_pipe_data(data_dir):
    ''' creates .xml's from all .dat files in data_dir. '''
    [generate_xml_for(filename, data_dir) for filename in os.listdir(data_dir)]
            
def generate_xml_for(filename, path):
    if not filename.endswith('.dat'): continue
    fullname = os.path.join(path, filename)
    with open(fullname) as f:
        content = f.readlines()
        for label in content:
            if label == 'FittingProximity\r\n':
            elif label == 'InadequateOrIncorrectClamping\r\n':
            elif label == 'JointMisaligned':
            elif label == 'NoClampUsed\r\n':
            elif label == 'NoGroundSheet\r\n':
            elif label == 'NoInsertionDepthMarkings\r\n':
            elif label == 'NoPhotoOfJoint\r\n':
            elif label == 'NoVisibleEvidenceOfScrapingOrPeeling\r\n':
            elif label == 'NoVisibleHatchMarkings\r\n':
            elif label == 'Other\r\n':
            elif label == 'PhotoDoesNotShowEnoughOfClamps\r\n':
            elif label == 'PhotoDoesNotShowEnoughOfScrapeZones\r\n':
            elif label == 'PoorPhoto\r\n':
            elif label == 'SoilContaminationHighRisk\r\n':
            elif label == 'SoilContaminationLowRisk\r\n':
            elif label == 'SoilContaminationRisk\r\n':
            elif label == 'UnsuitableScrapingOrPeeling\r\n':
            elif label == 'WaterContaminationHighRisk\r\n':
            elif label == 'WaterContaminationLowRisk\r\n':
            elif label == 'WaterContaminationRisk\r\n':
            else: print 'label %s in file %s not recognised' % 
                        (label, filename)


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
    if sys.argv[1] == 'generate_xml_labels_from_pipe_data':
        generate_xml_labels_from_pipe_data(sys.argv[2])
       
    else: print 'arg not recognised'
