
# ISSUES
# - once modify .bashrc, need to exit shell and restart shell!
# - script only works on lab machine - need path to gpu_sdk as command line argument! 

# Copy CUDA SDK to home directory
if [ -d ~/CUDA_SDK ]; then
    echo "deleting ~/CUDA_SDK ..."
    rm -r -f ~/CUDA_SDK
fi
mkdir ~/CUDA_SDK
echo "initialised empty directory ~/CUDA_SDK"
echo "copying /usr/local/cuda/gpu_sdk to ~/CUDA_SDK/ ..."
cp -r /usr/local/cuda/gpu_sdk ~/CUDA_SDK/
echo "copying /usr/local/cuda/samples to ~/CUDA_SDK/ ..."
cp -r /usr/local/cuda/samples ~/CUDA_SDK/


# Add env vars to ~/.bashrc file
echo "updating bash environment variables"
if ! grep -q 'export CUDA_BIN=/usr/local/cuda/bin' ~/.bashrc ; then
    echo 'export CUDA_BIN=/usr/local/cuda/bin' >>~/.bashrc
    export CUDA_BIN=/usr/local/cuda/bin
fi

if ! grep -q 'export CUDA_LIB=/usr/local/cuda/lib64' ~/.bashrc ; then
    echo 'export CUDA_LIB=/usr/local/cuda/lib64' >>~/.bashrc
    export CUDA_LIB=/usr/local/cuda/lib64
fi

if ! grep -q 'export LOCAL_BIN=${HOME}/.local/bin' ~/.bashrc ; then
    if grep -q 'LOCAL_BIN=' ~/.bashrc ; then
	echo ''
	echo 'ERROR: looks like you already have LOCAL_BIN as an environment variable in ~/.bashrc, but not in a format that can be recognised. please make sure the following string appears in ~/.bashrc'
	echo 'export LOCAL_BIN=${HOME}/.local/bin'
	echo 'and run this script again'
	exit
    fi
    echo 'export LOCAL_BIN=${HOME}/.local/bin' >>~/.bashrc
    export LOCAL_BIN=${HOME}/.local/bin
fi

if ! grep -q 'export LD_LIBRARY_PATH=${CUDA_LIB}:$LD_LIBRARY_PATH' ~/.bashrc ; then
    if grep -q 'LD_LIBRARY_PATH=' ~/.bashrc ; then
	echo ''
	echo 'ERROR: looks like you already have LD_LIBRARY_PATH as an environment variable in ~/.bashrc, but not in a format that can be recognised. please make sure the following string appears in ~/.bashrc'
	echo 'export LD_LIBRARY_PATH=${CUDA_LIB}:$LD_LIBRARY_PATH'
	echo 'and run this script again'
	exit
    fi
    echo 'export LD_LIBRARY_PATH=${CUDA_LIB}:$LD_LIBRARY_PATH' >>~/.bashrc
    export LD_LIBRARY_PATH=:${CUDA_LIB}:$LD_LIBRARY_PATH
fi

if ! grep -q 'export PATH=${CUDA_BIN}:${LOCAL_BIN}:$PATH' ~/.bashrc ; then
    if grep -q 'PATH=' ~/.bashrc ; then
	echo ''
	echo 'ERROR: looks like you already have PATH as an environment variable in ~/.bashrc, but not in a format that can be recognised. please make sure the following string appears in ~/.bashrc'
	echo 'export PATH=${CUDA_BIN}:${LOCAL_BIN}:$PATH'
	echo 'and run this script again'
	exit
    fi
    echo 'export PATH=${CUDA_BIN}:${LOCAL_BIN}:$PATH' >>~/.bashrc
    export PATH=${CUDA_BIN}:${LOCAL_BIN}:$PATH
fi
. ~/.bashrc
echo "updated bash environment variables"

# checking whether joblib, scikit-learn libraries installed; if not install them
python checkpylibs.py
if grep -q 'joblib' checkpylibs.txt; then
    echo "Downloading & installing joblib python library..."
    git clone https://github.com/joblib/joblib.git
    cd joblib
    python setup.py install --user
    cd ..
else
    echo "joblib is already installed"
fi
if grep -q 'sklearn' checkpylibs.txt; then

    echo "Downloading & installing scikit-learn (aka sklearn) python library..."
    git clone https://github.com/scikit-learn/scikit-learn.git
    cd scikit-learn
    python setup.py install --user
    cd ..
else
    echo "scikit-learn is already installed"
fi
rm -f checkpylibs.txt


# Fill in these environment variables.
# I have tested this code with CUDA 4.0, 4.1, and 4.2. 
# Only use Fermi-generation cards. Older cards won't work.

# If you're not sure what these paths should be, 
# you can use the find command to try to locate them.
# For example, NUMPY_INCLUDE_PATH contains the file
# arrayobject.h. So you can search for it like this:
# 
# find /usr -name arrayobject.h
# 
# (it'll almost certainly be under /usr)

# CUDA toolkit installation directory.
export CUDA_INSTALL_PATH=/usr/local/cuda

# CUDA SDK installation directory.
export CUDA_SDK_PATH=$HOME/CUDA_SDK

# Python include directory. This should contain the file Python.h, among others.
export PYTHON_INCLUDE_PATH=/usr/include/python2.7

# Numpy include directory. This should contain the file arrayobject.h, among others.
export NUMPY_INCLUDE_PATH=/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/

# ATLAS library directory. This should contain the file libcblas.so, among others.
export ATLAS_LIB_PATH=/usr/lib/atlas-base

make $*

# install DNouri's scripts
cd ../noccn
./setup.sh
cd -

echo ""
echo ""
echo "Congratulations! The beast of Cuda-Convnet augmented by Daniel Nouri's dropout and user-friendly scripts, as well as John B. McCormac's data-processing scripts are now installed."
echo "To get started, why not have fun training one of the most powerful neural nets in the world to recognise leaf-scan images better than any PlantClef competitor has ever done? Simply run the following command:"
echo "ccn-train models/basic_leafscan_network/options.cfg"
echo ""

bash
