import cPickle as pickle
import sys, os

if __name__ == '__main__':

  train_path = raw_input('path to train_output file? ')
  if not train_path.endswith('/'): train_path = train_path+'/'

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

  if pickle_it:
    pickle.dump(time_series, open(os.getcwd()+'/'+rootname+'.pickle','w'))

  if txt_it:
    data = open(train_path+'/'+rootname+'.txt','w')
    data.writelines(["%i\t%s\n" % (x,num) for x,num in enumerate(pretty_print)])
    data.close()

  if pickle_it or txt_it:
    print 'file(s) saved to pwd ie %s'%os.getcwd()
