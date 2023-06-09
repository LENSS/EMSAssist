# EMSAssist

EMSAssist is the first end-to-end low latency mobile voice assistant for Emergency Medical Services (EMS) at the edge. By taking the voice of EMS providers (e.g., emergency medical technician (EMT), volunteers) as input, EMSAssist outputs EMS protocols which prescribe appropriate medical interventions be administered to the patient. An accurate and fast protocol selection helps make a competent clinical decision regarding a treatment plan at emergency scenes. Thus, we aim to provide accurate and low-latency protocol selection assistance to EMTs with EMSAssist.

We design and deploy EMSAssist on mobile phones and on cloud. The end-to-end accuracy and latency evaluation show that: a) EMSAssist is more accurate than the state-of-the-art by a large margin, achieving a Top-5 accuracy above 96%; b) EMSAssist is the only EMS voice assistant that meets an service level objective (SLO) of 5 seconds latency on a mobile phone.

This repository contains the reproducible artifact for EMSAssist. We provide 3 ways to replicate the results in the EMSAssist paper: 1) replicate using a prebuilt docker image (recommended and provided); 2) replicate using a baremetal desktop/server machine with a decent NVIDIA GPU; 3) direct remote access (not available for now, we are moving the system to some public cloud for direct remote access in the near future, e.g., AWS). The neccessary text files for evaluating the accuracy of state-of-the-art protocol selection work are also provided in this repository.

To evaluate our EMSAssist artifact, downloading [reduced data.tar.gz (1.4GB, recommended)](https://drive.google.com/file/d/1ir-RBDJhf-wVFNTpsg64IqZ291XocVkb/view?usp=share_link) and [model.tar.gz (20GB)](https://drive.google.com/file/d/12LOuUl__T-oVMBQRLd8p7m27AiepQrSR/view?usp=sharing) are required. The compressed data file contains the audio input data for speech recognition evaluation, EMS text data for protocol selection selection evaluation. The compressed model file contains all pretrained/fine-tuned tensorflow models we developed. All the data and models are needed to replicate results in the EMSAssist paper. More details regarding the downloading are provided in Section 2.4 below.

## 1. Basic requirement

EMSAssist artifact evaluation relies on some basic software and hardware environment as shown below. The environment versions listed here are the versions we have tested. Different versions with small gaps should work (e.g., Ubuntu Desktop 18 and NVIDIA 3080 should work). It's good to note the artifact evaluation does not neccessarily need an NVIDIA GPU, but a NVIDIA GPU will help a lot for the evaluation (e.g., huge evaluation time reduction). In the following, we assume you have a least 1 NVIDIA GPU.

| Software Environment  | Version |
| ------------- | ------------- |
| OS  | Ubuntu Server 20.04.1 LTS |
| NVIDIA Driver  | 525.85.12  |

| Hardware Environment  | Version |
| ------------- | ------------- |
| GPU  | 3 x NVIDIA A30  |
| CPU | 2 x Intel Xeon 4314 |
| Disk | require about 75 GB |
| RAM | no specific requirement, suggest more than 10GB |

Before evaluating and using the EMSAssist artifact, please make sure you have at least 1 NVIDIA GPU available with `nvidia-smi` command. If you have lower-end GPU devices or no GPU device, no worry. Please start testing with the commands that do not contain `--cuda_device` flag and commands with `--cuda_device` flags set to be `-1`. Those 2 kinds of commands only require CPUs. We expect commands with `--cuda_device 0` are still functional on low-end GPU or on a machine without GPU. The difference should only be the execution time.

![nvidia-gpu](./nvidia-smi.png)


## 2. Replicate using a prebuilt docker image (recommended and provided)

The prebuilt docker image contains the neccessary software environment. We recommend using this option for the artifact evaluation. If you want to build the docker image on your own, please refer to the `build_docker.md` file in this repository.

Assuming NVIDIA GPUs present in the bare metal system running Ubuntu 22.04, we can start with artifact evaluation using prebuilt docker image. Again, if you have low-end GPU or have no GPU, you can still reproduce the results in the paper. Just wait for some commands to finish if your GPU device is not high-end enough.

As kindly suggested by our anounymous artifact evaluation reviewers, we provide a 2-hour video tutorial ([google drive video link](https://drive.google.com/drive/folders/14-UJsJXOJaZgTcoRq_6AqgUxhv_1ken4?usp=share_link)) to show how to fully evaluate our artifact. In the tutorial video we basically go through most of our commands in this repository. The outputs for each command are included. You may want to login into your google account first to view the video.

We first install docker and pull the docker image from dockerhub.

### 2.1 Install Docker on Bare Metal Machine

* Update and Install Docker:
```console
$ sudo apt update
$ sudo apt-get install docker
```

* Test Docker installation: (should show docker.service details.)
```console 
$ sudo systemctl status docker 
```
	
* Perform post installation steps to avoid sudo
```console
#Create the Docker group.
$ sudo groupadd docker

#Add your user to the Docker group
$ sudo usermod -aG docker $USER

#Activate the changes to groups
$ newgrp docker
```

### 2.2 Install Docker-Compose

```console
$ sudo apt update
$ sudo apt-get install docker-compose 
```

### 2.3 Install nvidia container tootlkit in local machine 

This toolkit allows you to connect the container engine to the bare metal machine's nvidia driver. This requires Docker dameon to reload.

```console
#enter the sudo user mode to add key and repository 
$ sudo -i
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu22.04/nvidia-docker.list > /etc/apt/sources.list.d/nvidia-docker.list
$ apt update
$ exit

#Install nvidia-container-toolkit
$ sudo apt -y install nvidia-container-toolkit

#Restart docker engine, this requires authentication
$ systemctl restart docker
```

### 2.4 Download the data and model

data: [reduced data.tar.gz (1.4GB, recommended)](https://drive.google.com/file/d/1ir-RBDJhf-wVFNTpsg64IqZ291XocVkb/view?usp=share_link), [data.tar.gz (140GB, will be deleted soon)](https://drive.google.com/file/d/1Li-oA6ZfuHx2EbqGWbhK-sZvwgnHVJs9/view?usp=sharing), [data1.tar.gz (140GB, will be deleted soon)](https://drive.google.com/file/d/1dsWkGsAbm0U1sNzsKYJMVtnyoAUo8koY/view?usp=sharing)

model: [model.tar.gz (20GB)](https://drive.google.com/file/d/12LOuUl__T-oVMBQRLd8p7m27AiepQrSR/view?usp=sharing), [model1.tar.gz (20GB)](https://drive.google.com/file/d/1zkWWY9624gMN2Qh8eNJcabps6MKDzY58/view?usp=sharing)

The tar.gz files above are on Google Drive now, we hope you access them after you have your Google Drive account login. We expect the downloading would take less than 30 minutes.

The 2 large data.tar.gz files may be deleted in the future. They contain lots of intermediate outputs during training/testing, which do not concretely help with artifact evaluation (e.g., tfrecord files, tensorflow model checkpoints during training). By removing those intermediate outputs, we are able to reduce the size of required data file from 140 GB to 1.4 GB. The results reproducibility is still maintained after the size reduction.

<!-- Additional Google Drive links for the ,  -->

### 2.5 Clone EMSAssist and decompress data into EMSAssist

```console
$ git clone --recursive https://github.com/LENSS/EMSAssist.git
$ cd EMSAssist

#Inside EMSAssist, decompress model.tar.gz
$ tar -xf model.tar.gz -C .

#Inside EMSAssist, decompress the data.tar.gz
$ tar -xf data.tar.gz -C .
```

<!-- ### 2.6 Download and decompress the data and model inside EMSAssist -->


<!-- the cuurent working (e.g., /home/$username/EMAssist) folder. We expect the downloading and decompressing to take 2-3 hours. -->

<!-- and [docker-compose.yml](https://drive.google.com/file/d/12LOuUl__T-oVMBQRLd8p7m27AiepQrSR/view?usp=share_link) -->

<!-- ```console



``` -->

With the steps above, we should have `data`, `model`, `init_models`, `examples`, `src`, `docker-compose.yml`， `requirements.txt` in `EMSAssist` folder.

### 2.6 Launch the docker

```console
#Run docker-compose in silent mode from EMSAssist folder
$ docker-compose up -d
```
It will pull a docker container image and run it in bare metal machine as "emsassist". The docker image size is about 20 GB.

### 2.7 Login the docker container and set up the data paths

<!-- $ conda activate emsassist-gpu -->

```console
$ docker exec -it emsassist /bin/bash

#Right now, we are in the `/home/EMSAssist` directory. Please make sure you can see nvidia-device after you login the docker
$ nvidia-smi

#The data path needs to be reset
$ cd data/transcription_text
$ python reconfig_data_path.py
$ cd ../..
```

We want to make sure the python path and library path are set up correctly (The two paths should already be set up).

$ echo $PYTHONPATH

> /home/EMSAssist/src/speech_recognition:/home/EMSAssist/examples

$ echo $LD_LIBRARY_PATH

> /root/anaconda3/envs/emsassist-gpu/lib:

<!-- ``` -->

<!-- Now you are inside the docker container as a sudo user, and your current location should be `root`. Before you go to specific `/home/EMSAssist/src` directories to evaluate the artifact, we want you to make sure the python path and library path are set up correctly (The two paths should already be set up). -->

<!-- ```console -->


<!-- * `echo $PYTHONPATH` -->

<!-- > /home/EMSAssist/src/speech_recognition:/home/EMSAssist/examples

* `echo $LD_LIBRARY_PATH`

> LD_LIBRARY_PATH=/opt/conda/envs/emsassist-gpu/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 -->

If, in some cases, the paths above do not match what's shown above inside the container, please set it up when you are in the `EMSAssist` folder while you login into the container:

```console
cd /home/EMSAssist
export PYTHONPATH=$PWD/src/speech_recognition:$PWD/examples
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
<!-- 
#correcting the data path, this is important
$ cd data/transcription_text/
$ python reconfig_data_path.py
$ cd ../.. -->



### 2.8 Begin the evaluation
```
#follow the README in the EMSAssist/src to continue the evaluation
$ cd src
```

## 3  Using bare metal machine 
First of all, we download anaconda for smoother artifact evaluation

* Download Anaconda installer: `wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh`

* Run the installer: `bash Anaconda3-2023.03-Linux-x86_64.sh`. Keep pressing `Enter` or inputing `yes` on the command line

* Create a conda environment for EMSAssist: `conda create -n emsassist-gpu pip python=3.7`

* Activate the environment: `conda activate emsassist-gpu`

* Install the XGBoost-GPU: `conda install py-xgboost-gpu`. This also installs the CudaToolkit: pkgs/main/linux-64::cudatoolkit-10.0.130-0

`conda install -c anaconda py-xgboost-gpu=0.90`

* Install the TensorFlow-2.9: `pip install tensorflow-gpu==2.9`

* Install the CUDA ToolKit 11.0 and CuDNN 8.0: `conda install -c conda-forge cudatoolkit=11.0 cudnn`

* Install the required python modules: `pip install -r requirements.txt`

### 3.1 Directory and path preparation

Before we proceed, please make sure you successfully set up the environment or get the Docker image running with `nvidia-smi`

* `git clone --recursive git@github.com:LENSS/EMSAssist.git`

* `cd EMSAssist`

* `git clone --recursive git@github.com:tensorflow/examples.git`

* `export PYTHONPATH=$PWD/src/speech_recognition:$PWD/examples`

* `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`

* Download the [data.tar.gz](https://drive.google.com/file/d/1dsWkGsAbm0U1sNzsKYJMVtnyoAUo8koY/view?usp=sharing) and [model.tar.gz](https://drive.google.com/file/d/1zkWWY9624gMN2Qh8eNJcabps6MKDzY58/view?usp=sharing) tar files from Google Drive to the cuurent EMSAssist folder. We expect the downloading and decompressing to take 2-3 hours.

* decompress the `model.tar.gz`: `tar -xvzf model.tar.gz`

* decompress the `data.tar.gz`: `tar -xvzf data.tar.gz`. After this step, make sure we have 5 folders under `EMSAssist` directory: `src`, `examples`, `data`, `init_models` and `model`.

* begin the evaluation: `cd src`. Check out the README file when you enter each sub-directory.

## 4 End-to-End Latency

To evaluate the end-to-end latency (i.e., Table 8 in the EMSAssist paper), we provide a [video](https://drive.google.com/file/d/1Yvxb4ppnlDOZyBa6Hom1bVoisBZkPKee/view?usp=share_link) with an [Android Studio log file](https://drive.google.com/file/d/1sVAR5Fmkn343qfcjF2-thuRxpiRtp1-X/view?usp=share_link). The video shows the real-time phone and android studio recording at the same time (end-to-end latency `4562` milliseconds).

<!-- The [phone_screen.mp4](https://drive.google.com/file/d/1NJJy2KaK8p0mjsarwKbk451PssdrE6-l/view?usp=share_link) shows how we launch the EMSAssist android application on our PH-1 mobile phone. The [android_studio_screen.mp4](https://drive.google.com/file/d/1Bi9ZhJf3OfLaGUJPpaZ5H2bf2ieKgy4n/view?usp=share_link) shows the real-time log (end-to-end latency `4549` milliseconds). -->

The source code of the android application is open source: https://github.com/liuyibox/EMSAssist-android. To try to install EMSAssist on your android phone, you may want to switch to branch `recording-button`.

To provide more practical analysis on the end-to-end analysis, we are working on evaluating more tflite models on mobile phones as shown in the todo table below. Please open an issue for feature request that may help you better understand and development on our EMSAssist.

- [x] EMSConformer + EMSMobileBERT
- [ ] EMSConformer + EMSMobileBERT (more mobile devices with different android)
- [ ] EMSConformer + ANN with one-hot on tokens (WIP)
- [ ] ContextNet + EMSMobileBERT
- [ ] ContextNet + ANN with one-hot on tokens
- [ ] RNNT + EMSMobileBERT
- [ ] RNNT + ANN with one-hot on tokens
- [ ] EMSConformer + EMSMobileBERT (iOS)


## 5 Notes

* This repo contains the testing commands. The training commands are updating.
* The current artifact can be evaluated with a bare-metal machine (prebuilt docker, and bare-metal configurations); We expect you may encounter some errors/issues when you want to evaluate EMSAssist on some non-bare metal machines (e.g., WSL, VM). If you have such needs, please file an issue for the feature request. We will see how we can help. This point is brought up by an anonymous artifact evaluation reviewer.
* All anonymous artifact reviewers and artifact shepherd contribute to the current shape of this artifact. The artifact credits also go to the reviewers and shepherd.
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
