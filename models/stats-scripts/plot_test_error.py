import cPickle as pickle
import sys, os, shutil, re
import numpy as np
import matplotlib.pyplot as plt
from math import ceil


def gnuplot(cfg_dir):
  os.system("gnuplot plot_test_error.gp")
  shutil.move(os.getcwd()+"/test_error_time_series.png",cfg_dir+"/test_error_time_series.png")
  os.remove(os.getcwd()+"/time_series.txt")

  
def write_data_to_txt(error):  
  data = open(os.getcwd()+'/time_series.txt','w')
  data.writelines(["%i\t%.5f \t%.5f\n" % (x,train,test)
                   for x,(train,test) in enumerate(error)])
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


def matplot(cfg_dir, error):
  x = np.array(range(len(error)))
  ytrain = np.array([train for (train,test) in error])
  ytest = np.array([test for (train,test) in error])
  plt.plot(x, ytrain)
  plt.plot(x, ytest)
  plt.xlabel('minibatch passes')
  plt.ylabel('error rate')
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

  matplot(cfg_dir, error)

  # gnuplot(cfg_dir)
