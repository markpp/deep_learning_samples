# Setting up development environment #
If you wish to use an Nvidia GPU, go to the buttom of the guide and begin with the CUDA guide. Otherwise start from the top and ignore the CUDA instructions.

## General tools and dependencies ##

Install and update essential system tools
```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install build-essential cmake cmake-gui g++ gfortran git pkg-config software-properties-common wget libatlas-dev libatlas-base-dev 
sudo apt-get install python-dev python2.7-dev python3.5-dev python-pip python-numpy python-scipy python-nose python-h5py python-matplotlib python-pandas python-sympy python-pygments python-sphinx python-setuptools python-cvxopt
```

In some cases, the python tools are not availble in newest versions from apt-get, then use pip instead. An exampel would be:
python-skimage. If python-skimage was installed using apt-get, you can do as follows to get a more recent version:
```bash
sudo pip uninstall scikit-image
sudo pip install scikit-image
```

# Python
Installing python packages using pip
```bash
sudo pip install --upgrade pip
sudo pip install pillow
sudo pip install scikit-image
sudo pip install scikit-learn
sudo pip install cython
sudo pip install pystruct
sudo pip install --user pyqpbo
```

# OpenCV
Get OpenCVs remaining dependencies
[ref](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)
```bash
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev
```

Build OpenCV from source along with the contrib module
```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv/
mkdir release
cd release/
cmake -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
sudo make -j7 install
```

# PCL
Get PCLs remaining dependencies
```bash
sudo apt-get install libeigen3-dev libvtk5-dev libusb-dev libgtest-dev git-core freeglut3-dev libxmu-dev libxi-dev libusb-1.0-0-dev graphviz mono-complete qt-sdk mpi-default-dev openmpi-bin openmpi-common libflann-dev libboost-all-dev
```

Build PCL from source
```bash
git clone https://github.com/PointCloudLibrary/pcl.git
cd pcl/
mkdir release
cd release/
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=ON -DBUILD_apps=ON -DBUILD_examples=ON ..
sudo make -j7 install
```

# Tensorflow
Begin by setting a TF environment variable according to your setup: [tensorflow](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#pip-installation)
```bash
# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Installing from sources" below.
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc0-cp27-none-linux_x86_64.whl
```

Then install using pip
```bash
sudo pip install --upgrade $TF_BINARY_URL
```

# Theano
```bash
sudo pip install Theano
```

(note) Switch between Theano using GPU or CPU by setting enviroment variable:
```python
Python example:
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
```

# Keras
[ref](http://www.pyimagesearch.com/2016/11/14/installing-keras-with-tensorflow-backend/)
```bash
sudo pip install keras
```

Configure keras to use Theano backend instead of Tensorflow
```bash
nano ~/.keras/keras.json
{
    "image_dim_ordering": "th", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "theano"
}
```

## Installing cuda 8.0 Ubuntu ##
[ref](https://github.com/saiprashanths/dl-setup)

# CUDA
Download CUDA 8.0 .deb file, for your system, from [nvidia-tools](https://developer.nvidia.com/cuda-toolkit)
```bash
sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```
Setting CUDA environment variables
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```
Restart
```bash
sudo shutdown -r now
```

# cuDNN
Login and download the cuDNN library that match your CUDA version through nvidias developer portal [nvidia-dev](https://developer.nvidia.com/cudnn)
```bash
cd ~/Downloads/
tar xvf cudnn*.tgz
cd cuda
sudo cp */*.h /usr/local/cuda/include/
sudo cp */libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

# Nvidia driver
NB: I recommend not installing the nvidia driver. Installing CUDA should take care of all you need. Here is how is can be done anyway.
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-367
```
