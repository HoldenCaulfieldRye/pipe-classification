import cPickle as pickle
import sys, os, shutil

if __name__ == '__main__':

  train_path = sys.argv[1]
  cfg_dir, train_output_fname = os.path.split(os.path.normpath(train_path))

  time_series = []
  pretty_print = []
  prev_testoutput = False
  with open(train_path) as f:
    content = f.readlines()
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

  data = open(os.getcwd()+'/time_series.txt','w')
  data.writelines(["%i\t%s\n" % (x,num) for x,num in enumerate(pretty_print)])
  data.close()

  os.system("gnuplot plot_test_error.gp")
  shutil.move(os.getcwd()+"/test_error_time_series.png",cfg_dir+"/test_error_time_series.png")
  os.remove(os.getcwd()+"/time_series.txt")

