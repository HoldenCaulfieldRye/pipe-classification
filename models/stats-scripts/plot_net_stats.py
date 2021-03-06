import cPickle as pickle
import sys, os, shutil, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import ceil
from subprocess import call
# from imp import load_path
# ccn = load_path('ccn', '../../noccn/noccn/ccn.py')  
# from ccn import shownet

def write_data_to_txt(error, num_epochs=-1):
  if num_epochs == -1: end = len(error)
  else: end = num_epochs*800
  data = open(os.getcwd()+'/time_series.txt','w')
  data.writelines(["%i\t%.5f \t%.5f\n" % (x,train,test)
                   for x,(train,test) in enumerate(error[0:end])])
  print "wrote txt data to %s"%(os.getcwd()+'/time_series.txt')
  data.close()

  
def parse(content):
  error = []
  test_error = 1 # blank would be better for matplotlib
  for idx in xrange(len(content)):

    if content[idx].startswith("Testing "):

      if content[idx].startswith("Testing frequency"):
        test_freq = int(content[idx].strip().split(' ')[-1])
        print "test_freq changed to %i"%(test_freq)

      else:
        for i in range(idx-test_freq-1,idx-1):
          try:
            assert content[i].strip().split(' ')[-1] == 'sec)'
            error.append((float(content[i].strip().split(' ')[3].split(',')[0]),test_error))
          except:
            # print 'ERROR: the following line was supposed to contain a train error:\n',content[i]
            # print 'So assume training was interrupted, so breaking'
            break
            
    if "===Test output===" in content[idx]:
      test_error = float(content[idx+1].split(',')[0].split('  ')[-1])

  for idx in xrange(len(content)):
    if content[idx].startswith("Saved checkpoint"):
      saved_net = '/'+content[idx].split('/',1)[1]

  return error, saved_net


def matplot(cfg_dir, error, start=-1, end=-1):
  
  if end == start == -1:
    start, end = 0, len(error)
    print 'plotting entire training data'
    
  elif start == -1:
    start = 0
    print 'plotting from epoch %i to %i'%(start,end)
    end *= 800
    
  elif end == -1:
    print 'plotting from epoch %i to the end'%(start)
    start, end = start*800, len(error)

  else:
    print 'plotting from epoch %i to %i'%(start,end)
    start, end = start*800, end*800
    
  x = np.array(range(len(error[start:end])))
  ytrain = np.array([train for (train,test) in error[start:end]])
  ytest = np.array([test for (train,test) in error[start:end]])
  plt.plot(x, ytrain, label='training error')
  plt.plot(x, ytest, label='validation error')
  plt.legend(loc='upper left')
  plt.xlabel('minibatch passes')
  plt.ylabel('logprob')
  # plt.title('Single-Input Logistic Sigmoid Neuron')
  plt.grid(True)
  plt.savefig(cfg_dir+"/plot_time_series_error_rates.png")
  print 'error time series plot saved in cfg_dir'
  # plt.show()


def make_filters(saved_net,layer_name,cfg_dir):
  back = os.getcwd()
  os.chdir('../../cuda_convnet')
  # dirty I know
  command = "python shownet.py -f "+saved_net+" --show-filter="+layer_name
  print 'about to call command: %s'%(command)
  call(command.split(), shell=False)
  rename = cfg_dir+'/filters_'+layer_name+'.png'
  print 'moving filters.png to %s'%(rename)
  shutil.move('filters.png', rename)
  os.chdir(back)


def make_preds(saved_net, cfg_dir, fail=False):
  back = os.getcwd()
  os.chdir('../../cuda_convnet')

  if fail==True:
    # dirty I know    
    command = "python shownet.py -f "+saved_net+" --show-preds=probs --only-errors=1"
    call(command.split(), shell=False)
    fail = 'fail_'

  else: 
    command = "python shownet.py -f "+saved_net+" --show-preds=probs"
    call(command.split(), shell=False)
    fail = ''

  preds_num = 0
  preds_dir = cfg_dir+'/preds'
  if not os.path.isdir(preds_dir):
    os.mkdir(preds_dir)
  while os.path.isfile('preds'+str(preds_num)+'.png'):
    shutil.move('preds'+str(preds_num)+'.png',
                preds_dir+'/'+fail+'preds'+str(preds_num)+'.png')
    preds_num += 1

  os.chdir(back)


if __name__ == '__main__':

  print "'--start-epoch=' or '--end-epoch=' accepted"

  # try: 
  #   os.environ['DISPLAY']

  # except: 
  #   print 'ERROR: X11 forwarding not enabled, cannot run script'

  # else:
  train_path = sys.argv[1]
  cfg_dir, train_output_fname = os.path.split(os.path.normpath(train_path))

  # to get absolute path of cfg_dir, not relative
  back = os.getcwd()
  os.chdir(cfg_dir)
  cfg_dir = os.getcwd()
  os.chdir(back)

  with open(train_path) as f:
    content = f.readlines()
    error, saved_net = parse(content)

  # can specify
  start,end = -1,-1
  for arg in sys.argv:
    if arg.startswith("--start-epoch="):
      start = int(arg.split('=')[-1])
    if arg.startswith("--end-epoch="):
      end = int(arg.split('=')[-1])

  matplot(cfg_dir, error, start, end)

  # ideal would be get layer names from cfg, and prompt for which ones
  # user wants
  if raw_input('create filters_conv1.png? ') == 'Y':
    make_filters(saved_net, 'conv1', cfg_dir)

  if raw_input('random preds[0-9].png? ') == 'Y':
    make_preds(saved_net, cfg_dir)
  if raw_input('fail preds[0-9].png? ') == 'Y':
    make_preds(saved_net, cfg_dir, True)


