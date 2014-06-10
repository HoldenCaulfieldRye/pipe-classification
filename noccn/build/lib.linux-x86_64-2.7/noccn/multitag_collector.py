# this script needs to return a nparray of (jpg_file_path, label) tuples. make sure they are in random order for good batch estimates. not sure what format label should be in for the multitagging, that depends on what cuda implementation can handle. but should you even do the multitagged training on cuda-convnet? no, caffe or torch should be better right?
 
# oh and collector(cfg) needs to return these labels. cfg is a dict of options.cfg so that's just how collector gets told which dir to look for jpgs
