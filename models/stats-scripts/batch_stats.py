import sys, os


def parse_command(sysargv):
  for param in [sutop, top, worst, suworst]:
    param = None

  for arg in sys.argv:
  if arg.startswith("--dir="):
    batch_dir = arg.split('=')[-1].split(',')
  if arg.startswith("--sutop="):
    sutop = arg.split('=')[-1].split(',')
  if arg.startswith("--top="):
    top = arg.split('=')[-1].split(',')
  if arg.startswith("--worst="):
    worst = arg.split('=')[-1].split(',')
  if arg.startswith("--suworst="):
    suworst = arg.split('=')[-1].split(',')

  return sutop, top, worst, suworst


def unpickle(listoflists):
  return [[pickle.load(open('data_batch_'+num)) for num in imglist]
          for imglist in listoflists]


def get_stats(batch_dir, sutop=None, top=None, worst=None, suworst=None):
    os.chdir(get_dir)

    listoflists = unpickle([sutop, top, worst, suworst])
    listoflists = unflatten()
    
    # pickle_label = open('batches.meta')



  
if __name__ == '__main__':

  print "eg: python batch_stats --dir=/data/batches --sutop==1,2 --top=3,4 --worst=5,6 --suworst=7,8"

  sutop, top, worst, suworst = parse_command(sys.argv)
  
  print "check: batch_dir: %s, sutop: %s, top: %s, worst: %s, suworst: %s"%(batch_dir,sutop,top,worst,suworst)
    
  get_stats(batch_dir, sutop, top, worst, suworst)
