import sys, os


def parse_command(sysargv):
  for param in [sutop, top, worst, suworst]:
    param = None
    
  batch_dir = os.path.abspath(sys.argv[1])
  for arg in sys.argv[2:]:
    if arg.startswith("--sutop="):
      sutop = arg.split('=')[-1].split(',')
    if arg.startswith("--top="):
      top = arg.split('=')[-1].split(',')
    if arg.startswith("--worst="):
      worst = arg.split('=')[-1].split(',')
    if arg.startswith("--suworst="):
      suworst = arg.split('=')[-1].split(',')

  return batch_dir, [sutop, top, worst, suworst]


def unpickle(fnum):
  # pickle_label = open('batches.meta')
  return pickle.load(open('data_batch_'+fnum))


def unflatten(batch):
  batch['data'] = batch['data'].reshape(batch['data'].shape[1],
                                        256, 256, 3) # shape[1]?
  batch['data'] = np.require(batch['data'],
                             dtype=np.uint8, requirements='W')
  return batch


def get_stats(batch_dir, dictlists):
  label_frequency = {}
  os.chdir(batch_dir)

  for imglist in dictlists.keys():
    dictlists[imglist] = [unpickle(fnum) for fnum
                          in dictlists[imglist]]
    # dictlists[imglist] = [unflatten(batch) for batch in imglist]
    label_frequency[imglist] = np.zeros(len(imglist[0]['labels']))
    for batch in label_frequency[imglist]:
      for label in batch['labels']:
        # assume batch['labels'][i] in {0,1,2,..,numclasses}
        label_frequency[imglist][label] += 1
    label_frequency[imglist] /= sum(label_frequency[imglist])

  return label_frequency


def store_imgs(dictlists, visual_inspect_dir=os.getcwd()):
  # create dirs in tree structure
  for root in dictlists:
    os.mkdir(visual_inspect_dir+'/'+root)
    
  


  
if __name__ == '__main__':

  print "eg: python batch_stats.py /data/batches --sutop==1,2 --top=3,4 --worst=5,6 --suworst=7,8"

  dictlists = {}
  batch_dir, [dictlists['sutop'], dictlists['top'], dictlists['worst'], dictlists['suworst']] = parse_command(sys.argv)
  
  print "check: batch_dir: %s, sutop: %s, top: %s, worst: %s, suworst: %s"%(batch_dir,dictlists['sutop'],dictlists['top'], dictlists['worst'], dictlists['suworst'])
    
  label_frequency = get_stats(batch_dir, dictlists)

  print "batch perf\t| clamp detected\t| no clamp\t| semi clamp"
  print "sutop\t| %s"  % (label_frequency['sutop'])
  print "top\t| %s"    % (label_frequency['top'])
  print "worst\t| %s"  % (label_frequency['worst'])
  print "suworst\t| %s"% (label_frequency['suworst'])

  # store_imgs(dictlists)
