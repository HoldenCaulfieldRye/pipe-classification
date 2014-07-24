import numpy as np
import os
from os.path import join as ojoin
from PIL import Image
import json, random

from create_lookup_txtfiles import create_lookup_txtfiles


# image-label mapping in data_info_dir/{train,val,test}.txt
def move_to_dirs(data_src_dir, data_dest_dir, data_info_dir):
  data_src_dir = os.path.abspath(data_src_dir)
  if not os.path.exists(data_dest_dir): os.mkdir(data_dest_dir)
  data_dest_dir = os.path.abspath(data_dest_dir)
  # task_dir = os.path.abspath(task_dir)

  train_dump,val_dump,test_dump = create_lookup_txtfiles(data_src_dir, data_info_dir)

  cross_val = [np.array(dump)
               for dump in [train_dump, val_dump, test_dump]]

  for dump,dname in zip(cross_val,['train','val','test']):
    dddir = ojoin(data_dest_dir,dname)
    os.mkdir(dddir)
    for fname in dump[:,0]:
      os.symlink(ojoin(data_src_dir,fname),ojoin(dddir,fname))

      
if __name__ == '__main__':
  import sys
  move_to_dirs(sys.argv[1], sys.argv[2])
