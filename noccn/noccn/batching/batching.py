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


#### STEP 2: VISUALLY INSPECT RANDOM SAMPLES OF DATA ##############

def sample_images(data_dir):
  sample_size = int(raw_input('How many images of each label do you want? '))
  d = get_label_dict(data_dir)
  d_small = {}
  for label in d.keys():
    d_small[label] = []
    if sample_size > len(d[label]):
      print 'there are only %i images with label %s'%(len(d[label]),label)
      d_small[label] = d[label]
    else: d_small[label] = random.sample(d[label], sample_size)
  whichBox = data_dir.split('/')[-3]
  json.dump(d, open('label_dict_sample_'+whichBox+'.txt','w'))      
  return d_small


def visual_inspect(data_dir):
  d = sample_images(data_dir)

  if list(data_dir)[-1] is not '/': data_dir = data_dir+'/'
  sample_dir = os.getcwd()+'/visual_inspect/'+data_dir.split('/')[-4]+'/'

  if os.path.isdir(os.getcwd()+'/visual_inspect'):
    os.chdir('visual_inspect')
  else: os.mkdir('visual_inspect')

  if os.path.isdir(sample_dir):
    rm = raw_input("image samples for inspection already found. delete? (Y/N) ")
    if rm == 'Y': 
      shutil.rmtree(sample_dir)
      os.mkdir(sample_dir)
  else:
    os.mkdir(sample_dir)
  os.chdir(sample_dir)

  for label in d.keys():
    inspect = raw_input("want to sample photos with %s? (Y/N) "%(label))
    if inspect == 'Y':
      if not os.path.isdir(sample_dir+label): os.mkdir(sample_dir+label)
      for filename in d[label]:
        if os.path.isfile(sample_dir+label+'/'+filename):
          print "have already sampled %s before"%(filename)
        else: 
          shutil.copyfile(data_dir+filename,sample_dir+label+'/'+filename)


#### STEP 3: (SKIP) LEAVE OUT BAD REDBOX DATA  #######################

def cleave_out_bad_data(data_dir):
  ''' creates 2 dirs, fills one with images in cwd having 
    'NoPhotoOfJoint' or 'PoorPhoto' as a label, fills other with the 
    rest. '''
  os.chdir(data_dir)
  good_data_dir = os.getcwd()+'/good_data/'
  bad_data_dir = os.getcwd()+'/bad_data/'
  os.mkdir(good_data_dir)
  os.mkdir(bad_data_dir)
  dirlist = os.listdir(data_dir)
  print 'looking through %i files...' % (len(dirlist))
  cleave_out_seq(data_dir,dirlist)

def cleave_out_par(data_dir,dirlist):
  ''' helper function for parallelisation. '''
  t = time.clock()
  Parallel(n_jobs=-1)(delayed(good_or_bad_train_case)(filename,
                                                         data_dir) 
                      for filename in dirlist if endswithdat(filename))
  print time.clock() - t, 'seconds to process cleave_out_par()'

def cleave_out_seq(data_dir,dirlist):
  ''' helper function for parallelisation. '''
  t = time.clock()
  [good_or_bad_train_case(filename,data_dir) for filename in 
   dirlist if endswithdat(filename)]
  print time.clock() - t, 'seconds to process cleave_out_seq()'

def endswithdat(filename):
  if filename.endswith('.dat'): return True
  return False 
  
def good_or_bad_train_case(filename,data_dir):
  ''' if file is .dat, see whether it contains a bad-training-case 
    label, if so create symlink to the .jpg (and the xml, the dat?) 
    inside bad_data_dir, otherwise inside good_data_dir. '''
  fullname = os.path.join(data_dir, filename)
  rootname = os.path.splitext(filename)[0]
  f = open(fullname)
  content = f.readlines()
  if 'NoPhotoOfJoint\r\n' in content or 'PoorPhoto\r\n' in content:
    os.symlink(fullname,data_dir+'/bad_data/'+rootname+'.jpg')
    os.symlink(fullname,data_dir+'/bad_data/'+rootname+'.dat')
  else: 
    os.symlink(fullname,data_dir+'/good_data/'+rootname+'.jpg')
    os.symlink(fullname,data_dir+'/good_data/'+rootname+'.dat')

# Parses a given .xml file, searching for the fields given by the
# list returns a dictionary of those fields, and their values in the
# file.
def get_info(fname,label_data_fields,metadata_file_ext):
  # metadata_file_ext is the file extension (eg .xml) for the data
  # file 
  fname = os.path.splitext(fname)[0] + metadata_file_ext # eg 'n012453' + '.xml'
  tree = ET.parse(fname)
  root = tree.getroot()
  return_dict = {}
  for label_data_field in label_data_fields:
    return_dict[label_data_field] = root.find(label_data_field).text
  return return_dict


#### STEP 4: (SKIP) CREATE XML DATA FILES IN CUDACONVNET FORMAT  #####
#### FOR TEST RUN AND FOR SERIOUS RUN                            #####

def generate_xml_labels_from_pipe_data(data_dir):
  ''' creates .xml's from all .dat files in data_dir. '''
  [generate_xml_for(filename, data_dir) for filename in os.listdir(data_dir)]
  
def generate_xml_for(filename, path):
  if not filename.endswith('.dat'): return
  rootname = os.path.splitext(filename)[0]
  fullname = os.path.join(path, filename)
  xmlname = os.path.join(path, rootname+'.xml')
  with open(fullname) as f:
    content = f.readlines()
    print 'content:', content
    data = {'labels':np.zeros(20,int),'bad_joint':np.zeros(1,int)}
    for label in content:
      if label == 'FittingProximity\r\n':
        data['labels'][0] = 1
      elif label == 'InadequateOrIncorrectClamping\r\n':
        data['labels'][1] = 1
      elif label == 'JointMisaligned':
        data['labels'][2] = 1
      elif label == 'NoClampUsed\r\n':
        data['labels'][3] = 1
      elif label == 'NoGroundSheet\r\n':
        data['labels'][4] = 1
      elif label == 'NoInsertionDepthMarkings\r\n':
        data['labels'][5] = 1
      elif label == 'NoPhotoOfJoint\r\n':
        data['labels'][6] = 1
      elif label == 'NoVisibleEvidenceOfScrapingOrPeeling\r\n':
        data['labels'][7] = 1
      elif label == 'NoVisibleHatchMarkings\r\n':
        data['labels'][8] = 1
      elif label == 'Other\r\n':
        data['labels'][9] = 1
      elif label == 'PhotoDoesNotShowEnoughOfClamps\r\n':
        data['labels'][10] = 1
      elif label == 'PhotoDoesNotShowEnoughOfScrapeZones\r\n':
        data['labels'][11] = 1
      elif label == 'PoorPhoto\r\n':
        data['labels'][12] = 1
      elif label == 'SoilContaminationHighRisk\r\n':
        data['labels'][13] = 1
      elif label == 'SoilContaminationLowRisk\r\n':
        data['labels'][14] = 1
      elif label == 'SoilContaminationRisk\r\n':
        data['labels'][15] = 1
      elif label == 'UnsuitableScrapingOrPeeling\r\n':
        data['labels'][16] = 1
      elif label == 'WaterContaminationHighRisk\r\n':
        data['labels'][17] = 1
      elif label == 'WaterContaminationLowRisk\r\n':
        data['labels'][18] = 1
      elif label == 'WaterContaminationRisk\r\n':
        data['labels'][19] = 1
      else: print 'label %s in file %s not recognised'%(label, filename)
      if 1 in [data['labels'][i] for i in [19,18,17,15,14,13]]:
        data['bad_joint'] = 1
    print 'dict ready:', data, ''
    # come_back = os.getcwd()
    # os.chdir(path)
    # xmlfile = open(xmlname, 'wb') # 'w' instead?
    # xmlcontents = dict2xml(data)
    # print 'xml ready:', xmlcontents.doc.toprettyxml(indent="  "), ''
    # xmlcontents.doc.writexml(xmlfile) 
    # xmlfile.close()
    print 'saved: %s'%(xmlname)
    # os.chdir(come_back)

    # PROBLEM: HOW TO SAVE XML FILE?
    # use Node.writexml() on the root node of your XML DOM tree.

    # problem: what is the root node of the xml dom tree of 
    # dict2xml(data) ?


#### STEP 5: STORE IMGS IN DIRS TO PREPARE FOR BATCHING  #############

def move_to_dirs(args):
  print 'you know what, this should be made into a ccn script, with arguments specified in options.cfg'
  print 'sys.argv[2] should be dir where raw data is'
  print 'sys.argv[3] should be dir in which to store labeled subdirs'
  print 'sys.argv[4] should be a string of the labels to lookup, separated by commas'
  print 'sys.argv[5] indicates that last label is the default one, eg for which no flag has been raised, if last_label_is_default is the arg value.'
  print 'CAREFUL: make sure your labels are spelled correctly! if they don\'t match those in data files, training cases won\'t be picked up correctly.'
  try: args[5]
  except: move_to_dirs_aux(args[2], args[3], args[4])
  else: 
    if args[5] == 'last_label_is_default':
      move_to_dirs_aux(args[2], args[3], args[4], True)
    else: print 'arg not recognised'
   
def move_to_dirs_aux(from_dir, to_dir, labels, lastLabelIsDefault=False):
  '''move_dir: where raw data is.
  to_dir: where to store labeled subdirs.
  labels: a string of the labels to lookup, separated by commas.
  lastLabelIsDefault: true iif last label is the default one, eg for
  which no flag has been raised.'''
  labels = labels.split(',') # all labels to train on
  list_dir = os.listdir(from_dir) # names of all elements in directory
  img_flags = [] # image's labels to train on
  case_count = 0 # number of training cases
  tagless_count = 0 # n
  badcase_count = 0 # number of images with multiple flags to train on

  # create label subdirs
  if not os.path.exists(to_dir):
    os.mkdir(to_dir)
  for label in labels:
    os.mkdir(to_dir+'/'+label.strip())

  # don't look up last label if it's a default one; it won't exist
  if lastLabelIsDefault: 
    default = labels[-1]
    del labels[-1]

  # create symlinks to images in appropriate dirs
  for filename in list_dir:
    if not filename.endswith('.dat'): continue
    case_count += 1
    fullname_dat = os.path.join(from_dir, filename)
    rootname = os.path.splitext(filename)[0]
    fullname_jpg = os.path.splitext(fullname_dat)[0]+'.jpg'
    with open(fullname_dat) as f:
      content = [line.strip() for line in f.readlines()] 
      img_flags = [label for label in labels if label in content]

      # if last label is a normal label, images with no labels will
      # not be batched
      if not img_flags: 
        if lastLabelIsDefault:
          os.symlink(fullname_jpg,to_dir+'/'+default+'/'+rootname+'.jpg')
        else: tagless_count += 1
      else:
        # if image has multiple flags, it will appear in each flag
        # subdir, each time with only one label. this is very bad for
        # training, so hopefully such cases are very rare.'
        if len(img_flags)>1: 
          badcase_count += len(img_flags)-1
          case_count += len(img_flags)-1
        for flag in img_flags:
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


#### STEP 5.1: SETUP IMBALANCE EXPERIMENT ############################

def imbalance_experiment(data_dir, min_ratio, max_ratio, num_nets):
  ''' given a data directory containing subdirs to each class, a range
  of imbalance ratios to cover, and a number of nets to train, creates
  num_nets directories, each holding a subdir for each class, with 
  max_ratio as imbalance for net_0, ..., min_ratio as imbalance for 
  net_num_nets. '''

  if min_ratio < 1 or max_ratio < 1: 
    print 'Error: ratios must be >=1.'
    exit

  # using cool log calculus, compute la raison de la suite 
  # geometrique donnant les ratios a obtenir pour chaque net.
  step = compute_step(min_ratio, max_ratio, num_nets)

  # move contents of data_dir to a new subdir, 'all'
  if os.path.isdir(ojoin(data_dir,'all')):
    shutil.rmtree(ojoin(data_dir,'all'))
  all_names = os.listdir(data_dir)
  for name in all_names:
    shutil.move(ojoin(data_dir,name), ojoin(data_dir,'all',name))

  # recursively make subdirs for each net, preserving strict set 
  # inclusion from net[i] to net[i+1]
  nets = ['all'] + ['net_'+str(i) for i in range(num_nets)]
  random_delete_recursive(data_dir, step, nets, ratio=2, i=0)
  print 'NOTE: net_0 has highest imbalance ratio.'


def random_delete_recursive(data_dir, step, nets, ratio, i):
#  os.mkdir(ojoin(data_dir,nets[i+1]))
  if os.path.isdir(ojoin(data_dir,nets[i+1])):
    shutil.rmtree(ojoin(data_dir,nets[i+1]))
  shutil.copytree(ojoin(data_dir, nets[i]), 
                  ojoin(data_dir, nets[i+1]), symlinks=True)
  random_delete_aux(ojoin(data_dir, nets[i+1]), ratio)
  if i+2 in range(len(nets)):
    random_delete_recursive(data_dir, step, nets, float(ratio)/step, i+1)


# careful! if you deleted links and now wish to add some back, make 
# sure json dump gets updated/overwritten correctly 
def random_delete_aux(data_dir, ratio):
  ''' randomly deletes as few images from outnumbering class dirs as
      possible such that #biggest/#smallest == ratio. '''

  data_dir = os.path.abspath(data_dir)
  dump = raw_input('Do you want a json dump in %s of which files were randomly deleted?(Y/any) '%(data_dir))
    
  # D is for dict, d is for directory
  D = {}
  os.chdir(data_dir)
  dirs = [d for d in os.listdir(data_dir) if os.path.isdir(ojoin(data_dir,d))]
  
  print 'the directories are: %s'%(dirs)

  for d in dirs:
    D[d] = {}
    D[d]['total'] = len(os.listdir(ojoin(data_dir,d)))

  dirs = [(d,D[d]['total']) for d in D.keys()]
  dirs = sorted(dirs, key = lambda x: x[1])

  print '%s is smallest class with %i images'%(dirs[0][0],dirs[0][1])
  for d in D.keys():
    D[d]['remove'] = max(0,int(D[d]['total']-(ratio*dirs[0][1])))
    print '%s has %i images so %i will be randomly removed'%(d, D[d]['total'], D[d]['remove'])
    if D[d]['remove'] > 0 :
      D = random_delete_aux2(data_dir,d,D)

  if dump == 'Y': json.dump(D, open(data_dir+'/random_remove_dict.txt','w'))
  return D


# D is for dict, d is for directory
def random_delete_aux2(data_dir,d,D,delete_hard=False):
  D[d]['deleted'] = random.sample(os.listdir(ojoin(data_dir,d)),D[d]['remove'])
  print 'successfully condemned images from %s'%(d)
  back = os.getcwd()
  os.chdir(ojoin(data_dir,d))
  for link in D[d]['deleted']: os.remove(link)
  os.chdir(back)
  return D


def compute_step(min_ratio, max_ratio, num_nets):
  ''' calculates step such that ratio[i+1] = ratio[i]*step for all i,   and such that '''
  return pow(max_ratio/float(min_ratio), 1/float(num_nets-1))


#### (SKIP) STEP 5.2: CREATE SHARED VALIDATION SET ###################

# SKIP: validation sets should respect proportions in training set.

# WARNING! this script probably has bugs, validation during training 
# was not working last time.


# to have flexible ratio, num_per_class must be num_min_class and code
# needs to change.
def separate_validation_set(data_dir, num_per_class=384, ratio=1):
  '''For comparing impact of different imbalance ratios: this script
  extracts a perfectly balanced validation set from a sequence of nets
  with strict training set inclusion. '''

  if os.path.exists(ojoin(data_dir,'validation')):
    shutil.rmtree(ojoin(data_dir,'validation'))
  nets = [net for net in os.listdir(data_dir) 
          if net.startswith('net_')]
  os.mkdir(ojoin(data_dir,'validation'))

  min_ratio_net_dir = ojoin(data_dir, 'net_'+str(len(nets)-1))
  print 'min_ratio_net_dir: %s'%('net_'+str(len(nets)-1))

  removed = extract_validation_set(min_ratio_net_dir, 
                                  num_per_class, ratio)

  for net in nets[:-1]: 
    remove_imgs(ojoin(data_dir,net), removed)

  print "Done. Now on each graphic machine, you need to:"
  print "  1) run batching on validation here on graphic02. "
  print "  2) scp the validation-batch dir and a net-raw dir from graphic02"
  print "  3) run batching on net dir"
  print "  4) copy validation-batch dir to each remote net-batch dir"
  print "  5) copy validation batches to net dir, but changing batch numbers such that they follow from the max batch in net dir. NOTE: you have a script for this :) merge_validation_batches()"


def batch_up(data_dir):
  '''batches up validation set, batches up each training set, merges
  validation batches into training sets. '''
  pass

# to have flexible ratio, num_per_class must be num_min_class and code
# needs to change.
def extract_validation_set(net_dir, num_per_class, ratio):
  '''randomly move num_per_class images out of each dir, and into a
  new sidealong dir called validation. '''
  classes = os.listdir(net_dir)
  print 'going to extract %i images from: %s'%(num_per_class,classes)
  d = {}
  for c in classes:
    os.mkdir(ojoin(net_dir, '..', 'validation',c))
    d[c] = random.sample(os.listdir(ojoin(net_dir,c)), num_per_class)
    for fname in d[c]:
      shutil.move(ojoin(net_dir, c, fname),
                  ojoin(net_dir, '..', 'validation',c))
  return d

def remove_imgs(net_dir, remove_dic):
  ''' remove_dics knows which imgs to remove in each class subdir,
  and does so in net_dir. '''
  for c in remove_dic.keys():
    for fname in remove_dic[c]:
      os.remove(ojoin(net_dir, c, fname))


def merge_validation_batches(data_dir):
  ''' assuming validation-batches dir is in the net-batches dir, moves
  contents of former into latter, but changing names of batches so
  that batch numbers follow sequentially and validation batch nums are
  highest. '''
  names = os.listdir(data_dir)
  train_batches = [name for name in names if name.startswith('data_batch_')]
  names = os.listdir(ojoin(data_dir, 'validation'))
  valid_batches = [name for name in names if name.startswith('data_batch_')]
  maxx = len(train_batches)

  for (i,batch) in enumerate(valid_batches):
    shutil.move(ojoin(data_dir,'validation',batch),
                ojoin(data_dir,'data_batch_'+str(maxx+i+1)))

  shutil.rmtree(ojoin(data_dir,'validation'))

  print 'validation batches start at data_batch_%i'%(maxx+1)
  print 'WARNING: batches.meta for validation thrown away. this might harm validation performance because the mean being subtracted will not be the mean over the validation set but over the training set. apart from that, don\'t think there\'s a problem. '



#### STEP 6: GENERATE BATCHES ########################################

# this isn't actually needed, just run ccn-make-batches
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

  # CAN IT WORK WITH SYMLINKS?

  # call a modified version of _collect_filenames_and_labels() from 
  # dataset.py, one that searches for .data files (and doesn't throw
  # error if no xml's found)

  # alternatively! convert .data files to xml in same format as for 
  # plant, and let john's scripts do the hard work.



#### SCRIPT ##########################################################

if __name__ == "__main__":
  import sys
  if sys.argv[1] == 'get_all_pipe_labels':
    get_all_pipe_labels(sys.argv[2])

  elif sys.argv[1] == 'get_label_dict':
    get_label_dict(sys.argv[2])

  elif sys.argv[1] == 'sample_images':
    sample_images(sys.argv[2])

  elif sys.argv[1] == 'visual_inspect':
    visual_inspect(sys.argv[2])

  elif sys.argv[1] == 'cleave_out_bad_data':
    cleave_out_bad_data(sys.argv[2])

  elif sys.argv[1] == 'doublecheck_cleave':
    get_all_pipe_labels('/data/ad6813/pipe-data/Redbox/good_data/',False)

  elif sys.argv[1] == 'generate_xml_labels_from_pipe_data':
    generate_xml_labels_from_pipe_data(sys.argv[2])

  # command used: python batching.py move_to_dirs /data2/ad6813/pipe-data/Bluebox/raw_data/dump /data2/ad6813/pipe-data/Bluebox/raw_data/clamp_detection NoClampUsed,PhotoDoesNotShowEnoughOfClamps,ClampDetected last_label_is_default

  elif sys.argv[1] == 'move_to_dirs':
    move_to_dirs(sys.argv)
    print 'WARNING: this script is BAD for multi-tagging'

  # and then: python batching.py imbalance_experiment /data2/ad6813/pipe-data/Bluebox/raw_data/clamp_detection

  # and then: nohup ~/.local/bin/ccn-make-batches models/clamp_detection/options.cfg >> models/clamp_detection/make_batches.out 2>&1 &
  elif sys.argv[1] == 'imbalance_experiment':
    print 'just need data_dir passed via command line'
    min_ratio = float(raw_input('min_ratio? '))
    max_ratio = float(raw_input('max_ratio? '))
    num_nets = int(raw_input('num_nets? '))
    imbalance_experiment(sys.argv[2], min_ratio, max_ratio, num_nets)

  # BUGGY! ignore because valid sets should respect proportions
  elif sys.argv[1] == 'separate_validation_set':
    num = int(raw_input('Number of class instances in validation set: (384) '))
    separate_validation_set(sys.argv[2], num)

  elif sys.argv[1] == 'merge_validation_batches':
    merge_validation_batches(sys.argv[2])

  else: print 'arg not recognised'
