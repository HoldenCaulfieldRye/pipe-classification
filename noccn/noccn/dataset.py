import cPickle as pickle
from fnmatch import fnmatch
import operator
import os
import random
import sys
import traceback
import math
import collections
import numpy as np
from PIL import Image
from PIL import ImageOps
from joblib import Parallel
from joblib import delayed
from script import get_options
from script import random_seed
from script import resolve
from ccn import mongoHelperFunctions
# This is used to parse the xml files
import xml.etree.ElementTree as ET # can be speeded up using lxml possibly

N_JOBS = -1


# Wrapper function which runs the process_item from
# a given instance of the BatchCreator class on an item.
# This is used for parallelizing the data.
def _process_item(creator, name):
    return creator.process_item(name)


# Yields chunks of a specified size n of a list until it
# is empty.  Chunks are not guaranteed to be of size n
# if the list is not a multiple of the chunk size
def chunks(l, n):
    for i in xrange(0, len(l), n):
        return_list = l[i:i+n]
        if len(return_list) == n:
            yield return_list


# The batch creator takes a list of files, and labels
# to produces numpy arrays based on their image data. 
# It also takes care of super set meta data, to allow
# for combining disparate classes at a later stage.
class BatchCreator(object):
    def __init__(self, batch_size=1000, channels=3, size=(256,256), output_path=None, 
                  n_jobs=N_JOBS, super_batch_meta=None, component=None, **kwargs):
        if output_path is None:
            print 'A valid output-path is required in the options file'
            sys.exit(1)
        self.setup_output_path(output_path)
        self.setup_super_meta(super_batch_meta)
        self.batch_size = batch_size
        self.channels = channels
        self.size = size
        self.n_jobs = n_jobs
        self.component = component
        vars(self).update(**kwargs)


    # Creates the output folder if it does not 
    # already exist, sets output path state
    def setup_output_path(self, output_path):    
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.output_path = output_path


    # Creates super meta file if it does not exist
    def setup_super_meta(self, super_batch_meta):    
        self.super_meta_filename = super_batch_meta
        if self.super_meta_filename is not None:
            if os.path.isfile(super_batch_meta):
                super_meta_file = open(self.super_meta_filename,'rb')
                self.super_meta = pickle.load(super_meta_file)
                super_meta_file.close()
            else:
                self.super_meta = { 'insert_list':{}, 'labels':{'super_labels':[]} }
    

    # Updates the super meta file with the label and insertion
    # information required for the combine step.  The insertion
    # algorithm creates a list of indices which need to have blanks
    # inserted, in order for each sub label probability array to be 
    # combined into a single super array.
    def update_super_meta(self, sorted_labels):
        if self.super_meta_filename is not None:
            self.super_meta['labels']['super_labels' ] += sorted_labels
            new_super = sorted(set(p for p in self.super_meta['labels']['super_labels' ]))
            self.super_meta['labels']['super_labels' ] = new_super
            self.super_meta['labels'][self.component] = sorted_labels
            for key in self.super_meta['labels']:
                if key != 'super_labels':
                    index = 0
                    insert_list = []
                    for label in self.super_meta['labels']['super_labels' ]:
                        if label in self.super_meta['labels'][key]:
                            index += 1
                            continue
                        else:
                            insert_list.append(index)
                    self.super_meta['insert_list'][key] = insert_list        
            super_meta_file = open(self.super_meta_filename,'wb')
            self.super_meta = pickle.dump(self.super_meta,super_meta_file)
            super_meta_file.close()
                

    # Takes a certain number of sample means from the batch
    def setup_batch_means(self, num_images, total_means = 20):    
        self.batches_per_mean = int((num_images/total_means)/self.batch_size)
        print 'Taking mean every %i batches'%(self.batches_per_mean)
        self.batch_means = None


    # Updates the mean batch and then stores the updated
    # information in the meta file
    def take_batch_mean(self, batch_num,  batch_data):
        if self.batches_per_mean <= 1 or batch_num % self.batches_per_mean == 0:
            print 'Taking mean of batch %i'%(batch_num)
            if self.batch_means is None:
                self.batch_means = batch_data.mean(axis=1).reshape(-1,1)
            else:    
                self.batch_means = np.hstack((self.batch_means,
                                            batch_data.mean(axis=1).reshape(-1,1)))
            # Save the new mean    
            self.save_batch_meta()


    # Creates the metadata file, with everything except a batch mean
    def setup_batch_meta(self, labels_sorted):
        self.batches_meta = {}
        self.batches_meta['label_names'] = labels_sorted


    # Writes the current meta file to disk, taking a current estimate
    # of the mean from the mean sampling system
    def save_batch_meta(self):
        self.batches_meta['data_mean'] = self.batch_means.mean(axis=1).reshape(-1,1)
        with open(os.path.join(self.output_path, 'batches.meta'), 'wb') as f:
            pickle.dump(self.batches_meta, f, -1)
            print 'Batch file backed_up'
            f.close()


    # The main method, goes through all the files, and batches it
    # while keeping track of the meta-data and super-meta data file
    # as specified in the configuration file.
    def __call__(self, all_names_and_labels, total_means = 200):
        # Setup super_meta data, and batch meta files including mean taking
        labels_sorted = sorted(set(p[1] for p in all_names_and_labels))
        self.update_super_meta(labels_sorted)
        self.setup_batch_meta(labels_sorted)
        self.setup_batch_means(len(all_names_and_labels))
        # Setup loop variables
        batch_num = 1
        for names_and_labels in list(chunks(all_names_and_labels,self.batch_size)):
            print 'Generating data_batch_%i'%(batch_num)
            rows = Parallel(n_jobs=self.n_jobs)(
                            delayed(_process_item)(self, name)
                            for name, label in names_and_labels)
            data = np.vstack([r for r in rows if r is not None])
            if data.shape[0] < self.batch_size:
                print 'Batch size too small, continuing to next batch'
                continue
            labels = np.array([labels_sorted.index(label) for ((name, label), row) 
                          in zip(names_and_labels, rows) if row is not None]).reshape((1,-1))
            batch = {'data': data.T, 'labels':labels, 'metadata': []}                         # data.T!! so dims are get_data_dimsxbatchSize
            self.take_batch_mean(batch_num,batch['data'])
            with open(os.path.join(self.output_path,'data_batch_%s'%batch_num),'wb') as f:
                pickle.dump(batch, f, -1)
                batch_num += 1
                f.close()
        print 'Batch processing complete'


    # Loads an image, and converts it to RGB format
    def load(self, name):
        return Image.open(name).convert("RGB")


    # Preserves aspect ratio, scales and crops an image to
    # size, converts it to a numpy array with 1D.  In row
    # major order, with [R G B] in that order.
    def preprocess(self, im):
        im = ImageOps.fit(im, self.size, Image.ANTIALIAS)
        im_data = np.array(im)
        im_data = im_data.T.reshape(self.channels, -1).reshape(-1)
        im_data = im_data.astype(np.single)
        return im_data


    # Try to process each image.  If it fails return None.
    def process_item(self, name):
        try:
            data = self.load(name)
            data = self.preprocess(data)
            return data
        except:
            print "Error processing batch, could not parse %s" % (name)
            return None


################################################################################
# XML Parsing Specific Functions
################################################################################
# Searches through a given directory for files matching the pattern,
# and returns those files
def find(root, pattern):
    for path, folders, files in os.walk(root, followlinks=True):
        for fname in files:
            if fnmatch(fname, pattern):
                yield os.path.join(path, fname)


# Parses a given .xml file, searching for the fields given by the list
# returns a dictionary of those fields, and their values in the file
def get_info(fname,label_data_fields,metadata_file_ext):
    fname = os.path.splitext(fname)[0] + metadata_file_ext
    tree = ET.parse(fname)
    root = tree.getroot()
    return_dict = {}
    for label_data_field in label_data_fields:
        return_dict[label_data_field] = root.find(label_data_field).text
    return return_dict


# Searches through a given directory for all of the .jpg and .xml
# files within it.  Parsing the contents according to the options.cfg
# The options.cfg is located in /models/XYZ/options.cfg
def _collect_filenames_and_labels(cfg):
    path = cfg['input-path']
    pattern = cfg.get('pattern', '*.jpg')
    metadata_file_ext = cfg.get('meta_data_file_ext', '.xml')
    label_data_field = cfg.get('label_data_field', 'ClassId')
    limit_by_tag = cfg.get('limit_by_tag', 'None')
    filenames_and_labels = []
    counter = 0
    if limit_by_tag == 'None':
        for fname in find(path, pattern):
            label = get_info(fname,[label_data_field],metadata_file_ext)[label_data_field]
            filenames_and_labels.append((fname, label))
            counter += 1
            sys.stdout.write("\rImages found: %d" % counter)
            sys.stdout.flush()
        print '\nNumber of images found: ',
    else:
        limit_to_tag = cfg.get('limit_to_tag', 'None')
        exclude = cfg.get('exclude', 'None')
        for fname in find(path, pattern):
            info_dict = get_info(fname,[limit_by_tag,label_data_field],metadata_file_ext)
            # This is to check whether it has an exclude set
            if ((limit_to_tag!='None' and info_dict[limit_by_tag]==limit_to_tag) or 
                   (exclude!='None' and info_dict[limit_by_tag]!=exclude) or 
                   (limit_to_tag == 'None' and exclude == 'None')):
                label = info_dict[label_data_field]
                filenames_and_labels.append((fname, label))
                counter += 1
                sys.stdout.write("\rImages found: %d" % counter)
                sys.stdout.flush()
        print '\nNumber of images of type ' + limit_to_tag + ' found: ',
    print len(filenames_and_labels)
    random.shuffle(filenames_and_labels)
    return np.array(filenames_and_labels)


################################################################################
# Console interpreter
################################################################################
def write_stats_to_file(path,labels):
    counter = collections.Counter(labels)
    stats_file = open(os.path.join(path, 'batch_stats.txt'), 'wb')
    stats_file.write('Number of label classes: %i \n'%(len(set(labels))))
    stats_file.write('Number of images: %i\n'%(len(labels)))
    stats_file.write(str(counter))
    stats_file.close()


def console():
    cfg = get_options(sys.argv[1], 'dataset')
    random_seed(int(cfg.get('seed', '42')))
    if int(cfg.get('xml_query',1)) != 0:
        collector = _collect_filenames_and_labels
        filter_component=str(cfg.get('limit_to_tag',None)).lower()
        filenames_and_labels = collector(cfg)
    else:
        images, labels = mongoHelperFunctions.bucketing(
                             threshold=int(cfg.get('class_image_thres',1000)),
                             component=cfg.get('limit_by_component',None),
                             componentProb=cfg.get('component_prob_thres',0.0),
                             )
        output_path=cfg.get('output-path', '/tmp/noccn-dataset')
        filter_component=str(cfg.get('limit_by_component',None)).lower()
        write_stats_to_file(output_path,labels)
        filenames_and_labels = zip(images,labels)
        random.shuffle(filenames_and_labels)
    creator = BatchCreator
    create = creator(
                 batch_size=int(cfg.get('batch-size', 1000)),
                 channels=int(cfg.get('channels', 3)),
                 size=eval(cfg.get('size', '(256, 256)')),
                 output_path=cfg.get('output-path', None),
                 super_batch_meta=cfg.get('super-meta-path', None),
                 component=filter_component,
                 )
    create(filenames_and_labels)


# Boilerplate for running the appropriate function.
if __name__ == "__main__":
    console()
