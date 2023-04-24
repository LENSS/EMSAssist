# EMSAssist

This repository contains the reproducible artifact for EMSAssist. We provide 3 ways to replicate the results in the EMSAssist paper: 1) replicate using a prebuilt docker image (recommended and provided); 2) replicate using a baremetal desktop/server machine with a decent NVIDIA GPU; 3) remote access (available upon request).

## 1. Basic requirement

EMSAssist artifact evaluation relies on some basic software and hardware environment as shown below. The environment versions listed here are the versions we have tested. Different versions with small gaps should work (e.g., Ubuntu Desktop 18 and NVIDIA 3080 should work). It's good to note the artifact evaluation does not neccessarily need an NVIDIA GPU, but a NVIDIA GPU will help a lot for the evaluation (e.g., huge evaluation time reduction). In the following, we assume you have a least 1 NVIDIA GPU.

| Software Environment  | Version |
| ------------- | ------------- |
| OS  | Ubuntu Server 20.04.1 LTS |
| NVIDIA Driver  | 525.85.12  |

| Hardware Environment  | Version |
| ------------- | ------------- |
| GPU  | 3 x NVIDIA A30   |
| CPU | 2 x Intel Xeon 4314 |
| Disk | require more than 200 GB |
| RAM | require more than 32GB |

Before evaluating and using the EMSAssist artifact, please make sure you have at least 1 NVIDIA GPU available with `nvidia-smi` command.

![nvidia-gpu](./nvidia-smi.png)

<!-- ## Build the target Environment

| Software Environment  | Version |
| ------------- | ------------- |
| OS  | Ubuntu Server 22.04.1 LTS |
| NVIDIA Driver  | 525.85.12  |
| CUDA Version  | 11   |
| CuDNN  | 8.0   |
| TensorFlow  | 2.9   | -->


## 2. Replicate using a prebuilt docker image (recommended and provided)

The prebuilt docker image contains the neccessary software environment. We recommend using this option for the artifact evaluation. The image can be built by following this DockerFile.

To start with artifact evaluation using prebuilt docker image, we first install docker and pull the docker image from dockerhub.

<!-- We follow the [official docker guide](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04) to install and run docker images: -->

Install docker:

Download the prebuilt docker image from dockerhub:

Launch a container with the prebuilt image: `docker-compose up -d`


## Using bare metal machine 
First of all, we download anaconda for smoother artifact evaluation

* Download Anaconda installer: `wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh`

* Run the installer: `bash Anaconda3-2023.03-Linux-x86_64.sh`. Keep pressing `Enter` or inputing `yes` on the command line

* Create a conda environment for EMSAssist: `conda create -n emsassist-gpu pip python=3.7`

* Activate the environment: `conda activate emsassist-gpu`

* Install the XGBoost-GPU: `conda install py-xgboost-gpu`. This also installs the CudaToolkit: pkgs/main/linux-64::cudatoolkit-10.0.130-0 

* Install the TensorFlow-2.9: `pip install tensorflow-gpu==2.9`

* Install the CUDA ToolKit 11.0 and CuDNN 8.0: `conda install -c conda-forge cudatoolkit=11.0 cudnn`

* Install the required python modules: `pip install -r requirements.txt`

## Directory and path preparation

Before we proceed, please make sure you successfully set up the environment or get the Docker image running with `nvidia-smi`

* `git clone --recursive git@github.com:LENSS/EMSAssist.git`

* `cd EMSAssist`

* `git clone --recursive git@github.com:tensorflow/examples.git`

* `export PYTHONPATH=$PWD/src/speech_recognition:$PWD/examples`

* `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`

* Download the [data.tar.gz](https://drive.google.com/file/d/1Li-oA6ZfuHx2EbqGWbhK-sZvwgnHVJs9/view?usp=share_link) and [model.tar.gz](https://drive.google.com/file/d/12LOuUl__T-oVMBQRLd8p7m27AiepQrSR/view?usp=share_link) tar files from Google Drive to the cuurent EMSAssist folder. We expect the downloading and decompressing to take 2-3 hours.

* decompress the `model.tar.gz`: `tar -xvzf model.tar.gz`

* decompress the `data.tar.gz`: `tar -xvzf data.tar.gz`. After this step, make sure we have 5 folders under `EMSAssist` directory: `src`, `examples`, `data`, `init_models` and `model`.

* begin the evaluation: `cd src`. Check out the README file when you enter each sub-directory.


<!-- we create and activate a conda environment with tensorflow-gpu: `conda activate tf-gpu` -->



<!-- ```
conda create -n xgb-gpu
conda activate xgb-gpu
conda install python=3.7
conda install py-xgboost-gpu
pip install tensorflow-gpu==2.9
```

`conda install -c conda-forge py-xgboost-gpu`

`mv /home/liuyi/anaconda3/lib/libstdc++.so.6.0.29 /home/liuyi/anaconda3/lib/libstdc++.so.6.0.29.old`

`ln -s /home/liuyi/anaconda3/envs/tf-gpu/lib/libstdc++.so.6.0.30 /home/liuyi/anaconda3/lib/libstdc++.so.6.0.29` -->
