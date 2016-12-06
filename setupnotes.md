# Notes on deep learning #


## TODO ##

Install and update essential system tools
```bash
sudo apt-get update  
sudo apt-get upgrade  
sudo apt-get install

sudo apt-get update && apt-get install build-essential cmake g++ gfortran git pkg-config software-properties-common wget python-dev python-pip python-numpy python-scipy python-nose python-h5py python-skimage python-matplotlib python-pandas python-sklearn python-sympy python-pygments python-sphinx python-setuptools python-cvxopt libatlas-dev libatlas3gf-base
```

## Installing cuda 8.0 Ubuntu ##

https://github.com/saiprashanths/dl-setup

nvidia driver
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-367
```

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
source ~/.bashrc
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


# Python
```bash
pip install pillow
pip install h5py
```


# Tensorflow
Begin by setting a TF environment variable according to your setup: [tensorflow](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#pip-installation)
```bash
# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Installing from sources" below.
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc0-cp27-none-linux_x86_64.whl
```

Then install using pip-installation
```bash
sudo pip install --upgrade $TF_BINARY_URL
```
# Theano
```bash
sudo pip install Theano
```

# Keras
http://www.pyimagesearch.com/2016/11/14/installing-keras-with-tensorflow-backend/
```bash
sudo pip install keras
```

Configure keras
```bash
nano ~/.keras/keras.json
```

# OpenCV
http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/
