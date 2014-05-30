import numpy as np
import os
import cPickle as pickle
from dict2xml import *
import pp
import xml.etree.ElementTree as ET
import shutil


def get_a_batch_data_array():
  ''' loads an array of labels in the format expected by cuda-
  convnet, from a hard-coded location which on graphic02 
  corresponds to a suitable data file. (batches.meta?) '''

def get_a_pipe_data_list(data_dir):
  ''' loads the 10002.data file provided by ControlPoint and stores
  its contents in a list, to be returned. '''
  os.chdir(os.cwd()+data_dir)


#### STEP 1: GET LABELS ##############################################

def get_all_pipe_labels(data_dir):
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
          d['labels'].sort()
          d['no_labels'] = len(d['labels'])
          pickle.dump(d, open('labels'+whichBox+'.pickle', 'wb'))
          print 'saved pickle file in', os.getcwd()


#### STEP 2: LEAVE OUT BAD REDBOX DATA  #############################

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

  # parallelisation tingz
  job_server = pp.Server()
  job1 = job_server.submit(cleave_out, (data_dir,dirlist,), (endswithdat,), (,))

  # cleave_out_bad_data_aux(data_dir,good_data_dir,bad_data_dir)

def cleave_out(data_dir,dirlist):
  ''' helper function for parallelisation. '''
  [good_or_bad_training_case(filename,data_dir) for filename in 
   dirlist if endswithdat(filename)]

def endswithdat(filename):
  if filename.endswith('.dat'): return True
  return False 
  
def good_or_bad_training_case(filename,data_dir):
  ''' if file is .dat, see whether it contains a bad-training-case 
    label, if so create symlink to the .jpg (and the xml, the dat?) 
    inside bad_data_dir, otherwise inside good_data_dir. '''
  fullname = os.path.join(data_dir, filename)
  rootname = os.path.splitext(filename)[0]
  with open(fullname) as f:
    content = f.readlines()
    if 'NoPhotoOfJoint\r\n' in content or 'PoorPhoto\r\n' in content:
      os.symlink(fullname,data_dir+'/bad_data/'+rootname+'.jpg')
    else: os.symlink(fullname,good_data_dir+'/good_data/'+rootname+'.jpg')

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


#### STEP 3: CREATE XML DATA FILES IN CUDACONVNET FORMAT  ############
#### FOR TEST RUN AND FOR SERIOUS RUN                     ############

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
        with open(xmlname,'w') as xmlfile:
          xmlfile = dict2xml(data)


#### STEP 4: GENERATE BATCHES ########################################

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
  good_or_bad_training_case('100002.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_training_case('100003.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_training_case('100004.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_training_case('100005.dat',data_dir,bad_data_dir,bad_data_dir)        
  good_or_bad_training_case('100006.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_training_case('100007.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_training_case('100008.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_training_case('100009.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_training_case('100010.dat',data_dir,bad_data_dir,bad_data_dir)
  good_or_bad_training_case('100011.dat',data_dir,bad_data_dir,bad_data_dir)
  if os.listdir(good_data_dir) == ['10000.dat','10000.dat','10000.dat','10000.dat','10000.dat','10000.dat','10000.dat','10000.dat'] and os.listdir(bad_data_dir) == ['100004.dat','100009.dat']: print 'test passed'
  else: print 'test failed.\nbad_data_dir contains:',os.listdir(bad_data_dir),'\nshould contain:',['100004.dat','100009.dat']
  shutil.rmtree(good_data_dir)
  shutil.rmtree(bad_data_dir)

def test_generate_xml_for():
  generate_xml_for('100002.dat',
                   '/data/ad6813/pipe-data/Redbox/')
  d = pipe_dataset.get_info('100002.jpg',['labels'],'.xml')
  if d == {'labels':np.array(
      [0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],int),
           'bad_joint':0}: 
    print '1 test passed, but make more!'
  else: 
    print 'test failed.\n dict:', d, '\nshould be:',{'labels':np.array([0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],int),'bad_joint':0}



#### SCRIPT ##########################################################

if __name__ == "__main__":
  import sys
  if sys.argv[1] == 'get_all_pipe_labels':
    get_all_pipe_labels(sys.argv[2])

  elif sys.argv[1] == 'cleave_out_bad_data':
    cleave_out_bad_data(sys.argv[2])

  elif sys.argv[1] == 'generate_xml_labels_from_pipe_data':
    generate_xml_labels_from_pipe_data(sys.argv[2])

  elif sys.argv[1] == 'test_cleave_out_bad_data':
    test_cleave_out_bad_data()

  elif sys.argv[1] == 'test_generate_xml_for':
    test_generate_xml_for()

  else: print 'arg not recognised'

