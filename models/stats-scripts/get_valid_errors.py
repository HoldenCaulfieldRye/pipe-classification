import cPickle as pickle

if __name__ == '__main__':

  print 'sys.argv[1]: path to train_output file'
  print 'sys.argv[2]: desired name of pickle file in which to store time series'

  time_series = []
  prev_testoutput = False
  with open(sys.argv[1]) as f:
    content = f.readlines()
    for line in content:
      if prev_testoutput:
        strnum = line.split(',')[0]
        num = float(strnum.split('  ')[-1])
        time_series.append(num)
        prev_testoutput = False
      elif '===Test output===' in line: prev_testoutput = True
      continue

  print ', '.join(time_series)
  pickle.dump(time_series, open(sys.argv[2]+'.pickle','w'))
