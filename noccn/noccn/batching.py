import numpy as np
import os
import cPickle as pickle
# from dict2xml import *
import xml.dom
from joblib import Parallel, delayed
from PIL import Image
import xml.etree.ElementTree as ET
import json, random
import shutil
import time



#### STEP 1: GET LABELS ##############################################

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

  print 'types of case_count, badcase_count, tagless_count: %s, %s, %s'%(type(case_count), type(badcase_count), type(tagless_count))
  print 'move_to_dir complete. summary stats:'
  print 'badcase_freq: %0.2f' % (float(badcase_count) / case_count)
  print 'tagless_freq: %0.2f' % (float(tagless_count) / case_count)

  return case_count, badcase_count, tagless_count


#### STEP 5.1: RANDOM DELETE FOR BALANCED CLASSES ####################

def random_delete(data_dir, ratio):
  ''' randomly deletes as few images from outnumbering class dirs as
      possible such that #biggest/#smallest == ratio. '''

  if ratio < 1: 
    print 'Error: ratio must be >=1.'
    exit

  data_dir = os.path.abspath(data_dir)
  dump = raw_input('Do you want a json dump in %s of which files were randomly deleted?(Y/any) '%(data_dir))
    
  # D is for dict, d is for directory
  D = {}
  os.chdir(data_dir)
  dirs = [os.path.join(data_dir,o) for o in os.listdir(data_dir) 
          if os.path.isdir(os.path.join(data_dir,o))]
  
  print 'the directories are: %s'%(dirs)

  for d in dirs:
    D[d] = {}
    D[d]['total'] = len(os.listdir(d))
    print '%s has %i images'%(d,D[d]['total'])

  dirs = [(d,num) for (d,D[d]) in D.keys()]
  dirs = sorted(dirs, key = lambda x: x[1])

  print '%s is smallest class with %i images'%(dirs[0][0],dirs[0][1])
  for d in D.keys():
    D[d]['remove'] = D[d]['total']-(ratio*dirs[0][1])
    print '%s has %i images so %i will be randomly removed'%(d, D[d]['total'], D[d]['remove'])
    D = random_remove(d,D)

  if dump == 'Y': json.dump(D, open(data_dir+'/'+,'w'))
  return D

# remember which files were deleted! (to make easy to bring missing 
# ones back in later)


# D is for dict, d is for directory
def random_remove(d,D):


#### STEP 6: GENERATE BATCHES ########################################

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


#### TEST FUNCTIONS ##################################################

def test_cleave_out_bad_data():
  data_dir = '/data/ad6813/pipe-data/Redbox'
  os.chdir(data_dir)
  good_data_dir = os.getcwd()+'/good_data/'
  bad_data_dir = os.getcwd()+'/bad_data/'
  os.mkdir(good_data_dir)
  os.mkdir(bad_data_dir)
  good_or_bad_train_case('100002.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_train_case('100003.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_train_case('100004.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_train_case('100005.dat',data_dir,bad_data_dir,bad_data_dir)        
  good_or_bad_train_case('100006.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_train_case('100007.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_train_case('100008.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_train_case('100009.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_train_case('100010.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_train_case('100011.dat',data_dir,bad_data_dir,bad_data_dir)
  if os.listdir(good_data_dir) == ['10000.dat','10000.dat','10000.dat','10000.dat','10000.dat','10000.dat','10000.dat','10000.dat'] and os.listdir(bad_data_dir) == ['100004.dat','100009.dat']: print 'test passed'
  else: print 'test failed.\nbad_data_dir contains:',os.listdir(bad_data_dir),'\nshould contain:',['100004.dat','100009.dat']
  shutil.rmtree(good_data_dir)
  shutil.rmtree(bad_data_dir)

def test_generate_xml_for():
  generate_xml_for('100002.dat',
                   '/data/ad6813/pipe-data/Redbox/')
  d = get_info('100002.jpg',['labels'],'.xml')
  if d == {'bad_joint':0,'labels':np.array(
      [0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],int)}: 
    print '1 test passed, but make more!'
  else: 
    print 'test failed.\n dict:', d, '\nshould be:',{'labels':np.array([0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],int),'bad_joint':0}

def test_move_to_dirs():
  labels = 'NoClampUsed,PhotoDoesNotShowEnoughOfClamps,ClampDetected'
  os.mkdir('temp_from')
  path_from = os.getcwd()+'/temp_from'
  path_to = os.getcwd()+'/temp_to'

  # create data files for the test
  base = os.getcwd()
  os.chdir(path_from)
  f1 = open('1.dat', 'a')
  f1.write('JointMisaligned\r\nNoInsertionDepthMarkings\r\n')
  f2 = open('2.dat', 'a')
  f2.write('')
  f3 = open('3.dat', 'a')
  f3.write('FittingProximity\r\nNoClampUsed\r\n')
  f4 = open('4.dat', 'a')
  f4.write('PhotoDoesNotShowEnoughOfClamps\r\n')
  f5 = open('5.dat', 'a')
  f5.write('NoClampUsed\r\nPhotoDoesNotShowEnoughOfClamps\r\n')
  f6 = open('6.dat', 'a')
  f6.write('NoClampUsed\r\n\NoGroundSheetr\nPhotoDoesNotShowEnoughOfClamps\r\n')
  f1.close()
  f2.close()
  f3.close()
  f4.close()
  f5.close()
  f6.close()
  for img_name in xrange(1,7):
    name = os.getcwd()+'/'+str(img_name)+'.jpg'
    shutil.copy('/data/ad6813/pipe-data/Redbox/100002.jpg',name)
    print 'copied a real jpg to %s'%(name)
  os.chdir(base)

  # run move_to_dirs on it
  summary_stats = move_to_dirs_aux(path_from,path_to,labels,True)
  
  # assimilate: subdir creation
  to_dirlist = os.listdir(path_to)
  if to_dirlist == ['NoClampUsed','PhotoDoesNotShowEnoughOfClamps','ClampDetected']: print 'labelled subdir creation: OK'
  else: 
    print 'labelled subdir creation INCORRECT'
    print 'to_dir: %s' % (to_dirlist)

  # assimilate: subdir populating
  noClamp_dirlist = os.listdir(path_to+'/'+to_dirlist[0])
  semiClamp_dirlist = os.listdir(path_to+'/'+to_dirlist[1])
  yesClamp_dirlist = os.listdir(path_to+'/'+to_dirlist[2])
  if noClamp_dirlist == ['3.jpg','5.jpg','6.jpg']:
    print 'noClamp subdir populating: OK'
  else: 
    print 'noClamp subdir populating INCORRECT'
    print 'is: %s\nshould be: %s'%(noClamp_dirlist,['3.jpg','5.jpg','6.jpg'])
  if semiClamp_dirlist == ['4.jpg','5.jpg','6.jpg']:
    print 'semiClamp subdir populating: OK'
  else: 
    print 'semiClamp subdir populating INCORRECT'
    print 'is: %s\nshould be: %s'%(semiClamp_dirlist,['2.jpg','5.jpg','6.jpg'])
  if yesClamp_dirlist == ['1.jpg','2.jpg']:
    print 'yesClamp subdir populating: OK'
  else: 
    print 'yesClamp subdir populating INCORRECT'
    print 'is: %s\nshould be: %s'%(yesClamp_dirlist,['1.jpg','2.jpg'])
  
  # assimilate: summary stats
  if summary_stats[0] == 8: print 'summary stats, case_count: OK'
  if summary_stats[1] == 2: print 'summary stats, badcase_count: OK'
  if summary_stats[2] == 0: print 'summary stats, tagless_count: OK'

  # make sure symlinked files are images
  try:
    img_link = path_to+'/'+to_dirlist[0]+'/'+noClamp_dirlist[0]
    if not os.path.islink(img_link): print '%s is not a link'%(img_link)
    img_name = os.readlink(img_link)
    Image.open(img_name).convert("RGB")
    print 'A file in one of the label subdirs links to a jpg image: OK'
  except: print 'ERROR: %s does not link to a jpg'%(img_link)

  # delete everything created by the test
  shutil.rmtree(path_from)
  shutil.rmtree(path_to)


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

  # command used: python batching.py move_to_dirs /data2/ad6813/pipe-data/Redbox/raw_data/dump /data2/ad6813/pipe-data/Redbox/raw_data/clamp_detection NoClampUsed,PhotoDoesNotShowEnoughOfClamps,ClampDetected last_label_is_default

  # and then nohup ~/.local/bin/ccn-make-batches models/clamp_detection/options.cfg >> models/clamp_detection/make_batches.out 2>&1 &
  elif sys.argv[1] == 'move_to_dirs':
    move_to_dirs(sys.argv)

##### tests ##########################################################

  elif sys.argv[1] == 'test_cleave_out_bad_data':
    test_cleave_out_bad_data()

  elif sys.argv[1] == 'test_generate_xml_for':
    test_generate_xml_for()

  elif sys.argv[1] == 'test_move_to_dirs':
    print '''WARNING: this test cannot make sure that the labels you 
          enter at command line will match those in data files.'''
    test_move_to_dirs()

  else: print 'arg not recognised'

