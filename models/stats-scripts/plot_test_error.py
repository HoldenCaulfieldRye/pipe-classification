import cPickle as pickle
import sys, os, shutil, re
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

  
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
            print 'ERROR: the following line was supposed to contain a train error:\n',content[i]
            print 'So assume training was interrupted, so breaking'
            break
            
    if "===Test output===" in content[idx]:
      test_error = float(content[idx+1].split(',')[0].split('  ')[-1])

  return error      


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
  plt.ylabel('% error rate')
  # plt.title('Single-Input Logistic Sigmoid Neuron')
  plt.grid(True)
  plt.savefig(cfg_dir+"/plot_time_series_error_rates.png")
  # plt.show()

  

if __name__ == '__main__':

  print "'--start-epoch=' or '--end-epoch=' accepted"

  train_path = sys.argv[1]
  cfg_dir, train_output_fname = os.path.split(os.path.normpath(train_path))

  with open(train_path) as f:
    content = f.readlines()
    error = parse(content)

  # can specify
  start,end = -1,-1
  for arg in sys.argv:
    if arg.startswith("--start-epoch="):
      start = int(arg.split('=')[-1])
    if arg.startswith("--end-epoch="):
      end = int(arg.split('=')[-1])
  
  matplot(cfg_dir, error, start, end)
    
  # gnuplot(cfg_dir)
