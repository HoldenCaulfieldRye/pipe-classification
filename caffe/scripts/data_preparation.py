
# n01440764/n01440764_10290.JPEG 0
# n01440764/n01440764_10293.JPEG 0

import numpy as np
import os
from os.path import join as ojoin
import cPickle as pickle
# from dict2xml import *
import xml.dom
from joblib import Parallel, delayed
from PIL import Image
import xml.etree.ElementTree as ET
import json, random
import shutil
import time


#### STEP 1: GET LABELS #############################################

def get_all_pipe_labels(data_dir,save=True):
  ''' looks into all .dat files in data_dir, and if find a new 
  label, add it to the list. stores final list as binary pickle 
  file.'''
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
  if save==True:
    d['labels'].sort()
    d['no_labels'] = len(d['labels'])
    pickle.dump(d, open('labels_'+whichBox+'.pickle', 'wb'))
    print 'saved pickle file in', os.getcwd()

def get_label_dict(data_dir):
  path = data_dir
  d = {'Perfect': []}
  print 'generating dict of label:files from %s...'%(data_dir)
  for filename in os.listdir(path):
    if not filename.endswith('.dat'): continue
    fullname = os.path.join(path, filename)
    with open(fullname) as f:
      content = f.readlines()
      if content == []:
        d['Perfect'].append(filename.split('.')[0]+'.jpg')
      else:
        for label in content:
          if label not in d.keys(): d[label] = []
          d[label].append(filename.split('.')[0]+'.jpg')
  return d



#### STEP 2: CREATE train.txt val.txt  #############################

def move_to_dirs(args):
  print 'you know what, this should be made into a ccn script, with arguments specified in options.cfg'
  print 'sys.argv[2] should be dir where raw data is'
  print 'sys.argv[3] should be dir in which to store .txt files'
  print 'sys.argv[4] should be a string of the labels to lookup, separated by commas'
  print 'sys.argv[5] indicates that last label is the default one, eg for which no flag has been raised, if last_label_is_default is the arg value.'
  print 'CAREFUL: make sure your labels are spelled correctly! if they don\'t match those in data files, training cases won\'t be picked up correctly.'
  try: args[5]
  except: move_to_dirs_aux(args[2], args[3], args[4])
  else: 
    if args[5] == 'last_label_is_default':
      move_to_dirs_aux(args[2], args[3], args[4], True)
    else: print 'arg not recognised'
   
def move_to_dirs_aux(data_dir, to_dir, labels):
  ''' data_dir: where raw data is
      to_dir: where to store .txt files
      labels: a string of the labels to lookup, separated by commas
      lastLabelIsDefault: true iif last label is the default one, eg 
  for which no flag has been raised '''
  label_dict = get_label_dict(data_dir)
  list_dir = os.listdir(from_dir) # names of all elements in dir
  lastLabelIsDefault = False
  img_labels = [] # image's labels to train on
  case_count = 0 # number of training cases
  tagless_count = 0 # n
  badcase_count = 0 # num of images with multiple flags to train on

  
  labels = label_dict.keys()
  for elem in enumerate(labels): print elem
  labels = [labels[int(num)] for num in raw_input("Numbers of labels to learn, separated by ' ': ").split()]

  if raw_input("Create default label for all images not containing any of given labels?") == 'Y':
    lastLabelIsDefault = True
    default = 'Default'
    
  # create symlinks to images in appropriate dirs
  for filename in list_dir:
    if not filename.endswith('.dat'): continue
    case_count += 1
    fullname_dat = os.path.join(from_dir, filename)
    rootname = os.path.splitext(filename)[0]
    fullname_jpg = os.path.splitext(fullname_dat)[0]+'.jpg'
    with open(fullname_dat) as f:
      content = [line.strip() for line in f.readlines()] 
      img_labels = [label for label in labels if label in content]

      # if last label is a normal label, images with no labels will
      # not be batched
      if not img_labels: 
        if lastLabelIsDefault:
          os.symlink(fullname_jpg,to_dir+'/'+default+'/'+rootname+'.jpg')
        else: tagless_count += 1
      else:
        # if image has multiple flags, it will appear in each flag
        # subdir, each time with only one label. this is very bad for
        # training, so hopefully such cases are very rare.'
        if len(img_labels)>1: 
          badcase_count += len(img_labels)-1
          case_count += len(img_labels)-1
        for flag in img_labels:
            os.symlink(fullname_jpg,to_dir+'/'+flag+'/'+rootname+'.jpg')

  print 'move_to_dir complete. summary stats:'
  print 'badcase_freq: %0.2f' % (float(badcase_count) / case_count)
  print 'tagless_freq: %0.2f' % (float(tagless_count) / case_count)

  labels.append(default)
  labels = merge_classes(to_dir, labels)
  labels = rename_classes(to_dir, labels)

  return case_count, badcase_count, tagless_count

def update_labels(labels, merge, new_label):
  labels = [label for label in labels if label not in [labels[i] for i in merge]]
  labels.append(new_label)
  return labels

def merge_classes(to_dir, labels):
  ''' once move_to_dirs is done, may wish to merge classes. '''
  more = 'Y'
  while more == 'Y':
    print '%s' % (', '.join(map(str,labels)))
    if raw_input('Merge (more) classes? (Y/N) ') == 'Y':
      merge = [-1]
      while not all([idx in range(len(labels)) for idx in merge]):
        for elem in enumerate(labels): print elem
        merge = [int(elem) for elem in raw_input("Name two class numbers from above, separated by ' ': ").split()]

      print 'moving files...'
      for fname in os.listdir(ojoin(to_dir,labels[merge[1]])):
        shutil.move(ojoin(to_dir,labels[merge[1]],fname),
                      ojoin(to_dir,labels[merge[0]]))
      new_label = raw_input('name of merged class? ')
      os.rmdir(ojoin(to_dir,labels[merge[1]]))
      os.rename(ojoin(to_dir,labels[merge[0]]), 
                ojoin(to_dir,new_label))
      labels = update_labels(labels, merge, new_label)

    else: more = False
  return labels

def rename_classes(to_dir, labels):
  ''' once move_to_dirs is done, may wish to rename classes (eg so 
  they can fit in preds). '''
  more = 'Y'
  while more == 'Y':
    if raw_input('Rename (another) class? (Y/N) ') == 'Y':
      rename = [-1]
      while not all([idx in range(len(labels)) for idx in rename]):
        for elem in enumerate(labels): print elem
        rename = [int(elem) for elem in raw_input("Name a class number from above: ").split()]
      new_name = raw_input('Rename to: ')
      os.rename(ojoin(to_dir,labels[rename[0]]), 
                ojoin(to_dir,new_name))
      labels = update_labels(labels, rename, new_name)
    else: more = 'N'
    return labels




if __name__ == '__main__':
  
