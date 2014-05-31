import cPickle as pickle
import os, sys
import convnet

def run(file_path):
    f = open(file_path,'rb')
    model_state = pickle.load(f)
    model_state['op'].print_usage()
    option_dict = vars(model_state['op'])
    for key in option_dict:
        print key
        print option_dict[key]['default']


if __name__ == '__main__':
    run(sys.argv[1])
