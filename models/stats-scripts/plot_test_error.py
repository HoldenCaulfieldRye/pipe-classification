import cPickle as pickle
import sys, os, shutil, re
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

  train_path = sys.argv[1]
  cfg_dir, train_output_fname = os.path.split(os.path.normpath(train_path))

  test_series = []
  prev_testoutput = False
  with open(train_path) as f:
    content = f.readlines()

    # get test error
    for line in content:
      if prev_testoutput:
        strnum = line.split(',')[0]
        strnum = strnum.split('  ')[-1]
        num = float(strnum)
        test_series.append(strnum)
        prev_testoutput = False
      elif '===Test output===' in line: prev_testoutput = True
      continue
      
    # get train error time averaged over two test error computations
    train_series = []
    for idx in xrange(len(content)):
      if content[idx].startswith("Testing "):
        if content[idx].startswith("Testing frequency"):
          test_freq = int(content[idx].strip().split(' ')[-1])
          print "test_freq changed to %i"%(test_freq)
        else:
          # print 'taking values from %i to %i'%(idx-2-test_freq,idx-1)
          # for j in range(idx-2-test_freq,idx-1): print content[j]
          # for i in range(idx-1-test_freq,idx-1):
          #   print len(content[i].strip().split(' ')) #[3]

          l = [float(content[i].strip().split(' ')[3].split(',')[0]) 
               for i in range(idx-test_freq-1,idx-1)]
          avg_train_error = reduce(lambda x,y: x+y, l) / len(l)
          train_series.append(str(avg_train_error))

    assert len(test_series) == len(train_series)
    pretty_print = zip(train_series, test_series)

  data = open(os.getcwd()+'/time_series.txt','w')
  data.writelines(["%i\t%s\t%s\n" % (x,train,test)
                   for x,(train,test) in enumerate(pretty_print)])
  print "wrote txt data to %s"%(os.getcwd()+'/time_series.txt')
  data.close()

  # os.system("gnuplot plot_test_error.gp")
  # shutil.move(os.getcwd()+"/test_error_time_series.png",cfg_dir+"/test_error_time_series.png")
  # os.remove(os.getcwd()+"/time_series.txt")

