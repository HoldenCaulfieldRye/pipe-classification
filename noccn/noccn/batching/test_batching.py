import numpy as np
import os
from os.path import join as ojoin
import cPickle as pickle
from joblib import Parallel, delayed
from PIL import Image
import json, random
import shutil
from batching import *


#### STEP 1: GET LABELS ##############################################

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


#### STEP 4: (SKIP) CREATE XML DATA FILES IN CUDACONVNET FORMAT  #####

def test_generate_xml_for():
  generate_xml_for('100002.dat',
                   '/data/ad6813/pipe-data/Redbox/')
  d = get_info('100002.jpg',['labels'],'.xml')
  if d == {'bad_joint':0,'labels':np.array(
      [0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],int)}: 
    print '1 test passed, but make more!'
  else: 
    print 'test failed.\n dict:', d, '\nshould be:',{'labels':np.array([0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],int),'bad_joint':0}


#### STEP 5: STORE IMGS IN DIRS TO PREPARE FOR BATCHING  #############

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
    shutil.copy('/data2/ad6813/pipe-data/Redbox/raw_data/dump/100002.jpg',name)
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

# test merge_classes?

# test rename_classes?


#### STEP 5.1: SETUP IMBALANCE EXPERIMENT #############################

def test_imbalance_experiment():
  os.mkdir('temp')
  os.mkdir(ojoin('temp','class1'))
  os.mkdir(ojoin('temp','class2'))
  
  # create 200 empty jpg's in class1, 20 in class2

  # setup imbalance experiment over temp directory

  # extract validation set of 10 imgs per class

  # verify that each jpg is now where it should be:
  # correct proportions in each net_i
  # correct strict inclusion from net_i+1 to net_i
  # correct number of files in validation
  # correct mutual exclusion of validation from all net_i

  # ---

  # batch everything up - ah but can't use empty jpgs for that

  # verify that 

#### STEP 5.2: CREATE VALIDATION SET #################################



#### SCRIPT ##########################################################



if __name__ == "__main__":
  import sys

  if sys.argv[1] == 'test_cleave_out_bad_data':
    test_cleave_out_bad_data()

  elif sys.argv[1] == 'test_generate_xml_for':
    test_generate_xml_for()

  elif sys.argv[1] == 'test_move_to_dirs':
    print '''WARNING: this test cannot make sure that the labels you 
          enter at command line will match those in data files.'''
    test_move_to_dirs()

  else: print 'arg not recognised'
