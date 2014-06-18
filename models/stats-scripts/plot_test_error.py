import cPickle as pickle
import sys, os, shutil, re

if __name__ == '__main__':

  train_path = sys.argv[1]
  cfg_dir, train_output_fname = os.path.split(os.path.normpath(train_path))

  time_series = []
  pretty_print = []
  prev_testoutput = False
  with open(train_path) as f:
    content = f.readlines()

    # get test error
    for line in content:
      if prev_testoutput:
        strnum = line.split(',')[0]
        strnum = strnum.split('  ')[-1]
        num = float(strnum)
        pretty_print.append(strnum)
        time_series.append(num)
        prev_testoutput = False
      elif '===Test output===' in line: prev_testoutput = True
      continue


    # get testing frequency
    test_freq = None
    for line in content:
      if line.startswith("Testing Frequency"):
        if test_freq not None:
          if line.strip().split(' ')[-1] not test_freq:
            print '''multiple test freqs found! time series will be 
            wrong, need to refine your code'''
            break
        test_freq = line.strip().split(' ')[-1]
        
      
    # get train error time averaged over two test error computations
    train_series = []
    next_trainoutput = False
    for idx in len(content):
      if content[idx].startswith("Testing "):
        if not content[idx].startswith("Testing Frequency"):
          l = [float(line.strip().split(' ')[3].split(',')[0]) 
               for line in content[idx-2-test_freq:idx-2]]
          avg_train_error = reduce(lambda x,y: x+y, l) / len(l)
          train_series.append(avg_train_error)
          
  data = open(os.getcwd()+'/time_series.txt','w')
  data.writelines(["%i\t%s\n" % (x,num) for x,num in enumerate(pretty_print)])
  data.close()

  os.system("gnuplot plot_test_error.gp")
  shutil.move(os.getcwd()+"/test_error_time_series.png",cfg_dir+"/test_error_time_series.png")
  os.remove(os.getcwd()+"/time_series.txt")

