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
          error.append((float(content[i].strip().split(' ')[3].split(',')[0]),
                        test_error))

    if "===Test output===" in content[idx]:
      test_error = float(content[idx+1].split(',')[0].split('  ')[-1])

  return error      


def matplot(cfg_dir, error, num_epochs=-1):
  if num_epochs == -1:
    end = len(error)
    print 'plotting entire training data'
  else:
    end = num_epochs*800
    print 'plotting %i epochs' % (num_epochs)
  x = np.array(range(len(error[:end])))
  ytrain = np.array([train for (train,test) in error[:end]])
  ytest = np.array([test for (train,test) in error[:end]])
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

  train_path = sys.argv[1]
  cfg_dir, train_output_fname = os.path.split(os.path.normpath(train_path))

  with open(train_path) as f:
    content = f.readlines()
    error = parse(content)

  try:
    matplot(cfg_dir, error, int(sys.argv[2]))
  except:
    matplot(cfg_dir, error)
    
  # gnuplot(cfg_dir)
