
# Setup for M3D-RPN

This document acts as a (suggested) guide for setting up cuda, Python 3, and Anaconda. If components are already setup please feel encouraged to skip sections or use any alternative methods such as pip. 

*Note:* there are MANY alternative methods to install all below packages. This guide is only meant to serve as an example. 

#### Install cuda 8

1. Visit [https://developer.nvidia.com/cuda-80-ga2-download-archive]([https://developer.nvidia.com/cuda-80-ga2-download-archive)

1. Download the Linux -> x86_64 -> Ubuntu -> 16.04 -> deb (local) file.
	
1. Then install by
	```
	cd <download folder>
	sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
	sudo apt-get update
	sudo apt-get install cuda-8-0
	```
	Dealing with multiple cuda versions?
	The default cuda is a softlink at */usr/local/cuda*

	If you want to change the default to cuda 8 follow the below lines, or update your environment variables. 
	```
	sudo rm /usr/local/cuda
	sudo ln /usr/local/cuda-8.0 /usr/local/cuda
    ```
	If you want to see the current default version: *ls -l /usr/local/*

#### Install proper cuDNN
		
For M3D-RPN we utilized cudnn-8.0-linux-x64-v5.1. However, please choose the appropriate package for your cuda. 

1. You must create an account to access this page: [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)

1. Then download your preferred version or use this link for [cudnn-8.0-linux-x64-v5.1](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod_20161129/8.0/cudnn-8.0-linux-x64-v5.1-tgz).

1. Extract somewhere temporary for example to downloads, e.g., *~/Downloads/cuda*.

1. Then copy the files into your cuda directory as below:
	```
	sudo cp ~/Downloads/cuda/include/* /usr/local/cuda-8.0/include/
	sudo cp ~/Downloads/cuda/lib64/* /usr/local/cuda-8.0/lib64/
	```

#### Install Anaconda / Python 3.6.5

For M3D-RPN we utilized Python 3.6.5 Anaconda. Other versions may also work.

1. Install your preferred version of Anaconda
	```
	cd ~
	wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
	sh Anaconda3-5.2.0-Linux-x86_64.sh
	```

	Defaults are usually fine. 
	Recommend letting the installer add to your path and avoid microsoft extention (unless on windows). 
	Before doing any of the below make sure that the path is setup properly:
	
    ```
    python --version
    ```
	Hopefully you see Python 3.6.5, Anaconda Inc. 

1. Install python packages.
	
	```
	conda install -c menpo opencv3=3.1.0 openblas
	conda install cython scikit-image h5py nose pandas protobuf atlas libgfortran
	```
    
    If there are compatibility issues. Refer to [python_packages.txt](python_packages.txt) for specific versions known to work. 

1. Install pytorch

	Assuming cuda-8.0 is installed. Otherwise, refer to the official [pytorch website](https://pytorch.org/). 

	```
	conda install pytorch torchvision cuda80 -c pytorch
	```

1. Install visdom (optional, for graph monitoring while training)
	```
	conda install -c conda-forge visdom
	```



