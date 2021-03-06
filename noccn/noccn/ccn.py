import sys, os, inspect


def add_path():
    path_from_main = "/homes/ad6813/Git/pipe-classification/cuda_convnet"
    # print path_from_main
    if os.path.isdir(path_from_main):
        sys.path.append(path_from_main)
    path_from_folder = "./cuda_convnet"
    if os.path.isdir(path_from_folder):
        sys.path.append(path_from_folder)
    path_from_main = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../bucketing")
    if os.path.isdir(path_from_main):
        sys.path.append(path_from_main)
    path_from_folder = "./bucketing"
    if os.path.isdir(path_from_folder):
        sys.path.append(path_from_folder)

add_path()


import convnet
import data
import gpumodel
import options
import shownet
