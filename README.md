pipe-classification
===================

Imperial College Individual Project: automatic classification of pipe weld images.



setup
=====

for python to find .so files, need their location to be added to LD_LIBRARY_PATH environment variable.
also, you must run this program in a bash terminal.
in bashrc, set:
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH



import pretrained net
=====================

under your options.cfg file, under [train], set 
f = /path/to/pretrained/net 



import DeCAF
============

clone the repo: https://github.com/UCB-ICSI-Vision-Group/decaf-release/
modify decaf/layers/cpp/Makefile: remove '-Wl' from l3
install dependencies:
MPI WITHOUT ROOT SOMEHOW!
pip install --install-option="--prefix=$HOME/.local" networkx
pip install --install-option="--prefix=$HOME/.local" mpi4py 
pip install --install-option="--prefix=$HOME/.local" numexpr
pip install --install-option="--prefix=$HOME/.local" scikit-image
pip install --install-option="--prefix=$HOME/.local" 
pip install --install-option="--prefix=$HOME/.local" 
pip install --install-option="--prefix=$HOME/.local" 
pip install --install-option="--prefix=$HOME/.local" 

upgrade boost (if you don't have v1.55):
wget http://downloads.sourceforge.net/project/boost/boost/1.55.0/boost_1_55_0.tar.gz
