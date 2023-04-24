# Build EMSAssist docker image from scratch

Build Docker image with all required packages. This step is optional as a ready image will be pulled in absence from Docker hub (step 7). This step involves manual installation of packages and rebuilding the image which will take some time.
	* Create Dockerfile
	```console
	touch Dockerfile
	```
	* Edit Dockerfile with your preffered text editor and paste the following contents in Dockerfile then save it.
	```dockerfile
	#FROM ubuntu:22.04
	FROM nvidia/cuda:12.0.1-base-ubuntu22.04
	MAINTAINER "Amran Haroon"

	ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
	ENV PATH /opt/conda/bin:$PATH

	RUN set -x && \
		apt-get update --fix-missing && \
		apt-get install -y --no-install-recommends \
			bzip2 \
			ca-certificates \
			git \
			libglib2.0-0 \
			libsm6 \
			libxcomposite1 \
			libxcursor1 \
			libxdamage1 \
			libxext6 \
			libxfixes3 \
			libxi6 \
			libxinerama1 \
			libxrandr2 \
			libxrender1 \
			mercurial \
			openssh-client \
			procps \
			subversion \
			wget \
		&& apt-get clean \
		&& rm -rf /var/lib/apt/lists/* && \
		UNAME_M="$(uname -m)" && \
		if [ "${UNAME_M}" = "x86_64" ]; then \
			ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh"; \
			SHA256SUM="19737d5c27b23a1d8740c5cb2414bf6253184ce745d0a912bb235a212a15e075"; \
		elif [ "${UNAME_M}" = "s390x" ]; then \
			ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-s390x.sh"; \
			SHA256SUM="f5ccc24aedab1f3f9cccf1945ca1061bee194fa42a212ec26425f3b77fdd943a"; \
		elif [ "${UNAME_M}" = "aarch64" ]; then \
			ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-aarch64.sh"; \
			SHA256SUM="fbadbfae5992a8c96af0a4621262080eea44e22baee2172e3dfb640f5cf8d22d"; \
		elif [ "${UNAME_M}" = "ppc64le" ]; then \
			ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-ppc64le.sh"; \
			SHA256SUM="8fdebc79f63b74daad421a2674d43299fa9c5007d85cf00e8dc1a81fbf2787e4"; \
		fi && \
		wget "${ANACONDA_URL}" -O anaconda.sh -q && \
		echo "${SHA256SUM} anaconda.sh" > shasum && \
		sha256sum --check --status shasum && \
		/bin/bash anaconda.sh -b -p /opt/conda && \
		rm anaconda.sh shasum && \
		ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
		echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
		echo "conda activate base" >> ~/.bashrc && \
		find /opt/conda/ -follow -type f -name '*.a' -delete && \
		find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
		/opt/conda/bin/conda clean -afy

	RUN set -x && \     
		conda create -yn emsassist-gpu pip python=3.7 

	#ENTRYPOINT ["tail", "-f", "/dev/null"]
	#CMD [ "/bin/bash" ]
	ENTRYPOINT ["tail"]
	CMD ["-f","/dev/null"]
	```

	* Build Docker image and give it a name:
	```console
	$ docker build -t haroon3rd/anaconda3:base .
	```
	
	* Make sure the image was created successfully
	```console
	$ docker image ls
	```
	
	* Run Docker image with gpu enabled in silent mode and execute into terminal :
	```console
	$ docker run --gpus all -d -t --name base haroon3rd/anaconda3:base
	$ docker exec -it base /bin/bash
	```
	
	*  Install the required packages inside created conda environment:
	```console
	# Activate conda env 
	$ conda activate emsassist-gpu
	
	# Install gcc
	$ apt-get update && apt-get -y install gcc mono-mcs && rm -rf /var/lib/apt/lists/*

	# Install the required python modules one by one
	$ pip install tensorflow_addons sentencepiece gin-config tflite-support tensorflow_hub natsort scikit-learn fire pyyaml tqdm librosa tensorflow-io==0.26 tensorflow_datasets nltk pydub pandas

	# Install the XGBoost-GPU
	$ conda install py-xgboost-gpu
	# This also installs the CudaToolkit: pkgs/main/linux-64::cudatoolkit-10.0.130-0
	
	# Install the TensorFlow-2.9
	$ pip install tensorflow-gpu==2.9
	
	# Install the CUDA ToolKit 11.0 and CuDNN 8.0
	$ conda install -c conda-forge cudatoolkit=11.0 cudnn
	```

	* Save this container (with changes) to a new image (may take up to 10 mins):
	```console
	$ docker commit base  haroon3rd/anaconda3:nvidia-v1
	```

	* Make sure you have a new image `haroon3rd/anaconda3:nvidia-v1`:
	```console
	$ docker image ls
	```

3. Install Docker-Compose
	```
    $ sudo apt update
	$ sudo apt-get install docker-compose 
    ```

4. install nvidia container tootlkit in local machine (requires Docker dameon to reload).
	This toolkit allows you to connect the container engine to the bare metal machine's nvidia driver.

	```
    # Add key and repository 
	$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
	$ curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu22.04/nvidia-docker.list > /etc/apt/sources.list.d/nvidia-docker.list
	
    # Update repository
	$ apt update
	
    # Install nvidia-container-toolkit
	$ sudo apt -y install nvidia-container-toolkit
	
    # Restart docker engine
	$ systemctl restart docker
    ```
5. Download the [data.tar.gz](https://drive.google.com/file/d/1Li-oA6ZfuHx2EbqGWbhK-sZvwgnHVJs9/view?usp=share_link), [model.tar.gz](https://drive.google.com/file/d/12LOuUl__T-oVMBQRLd8p7m27AiepQrSR/view?usp=share_link) and [docker-compose.yml](https://drive.google.com/file/d/12LOuUl__T-oVMBQRLd8p7m27AiepQrSR/view?usp=share_link) files from Google Drive to the cuurent working (i.e., ./home) folder. We expect the downloading and decompressing to take 2-3 hours.

    * decompress the `model.tar.gz`: `tar -xvzf model.tar.gz`

    * decompress the `data.tar.gz`: `tar -xvzf data.tar.gz`. After this step, make sure we have 4 items in the current directory: `data`, `model`, and `EMSAssist` folder and a file 'docker-compose.yml'. Also make sure there are 3 folders under `EMSAssist` directory: `src`, `examples`, and `init_models`.

	* Make sure `docker-compose.yml` has the following content in it:
	```yaml
	version: '3.7'

	services:
	  emsassist:
        image: haroon3rd/anaconda3:nvidia-v1
          container_name: emsassist
          volumes:
            - ./data:/home/EMSAssist-artifact-evaluation/data
            - ./model:/home/EMSAssist-artifact-evaluation/model
            - ./EMSAssist:/home/EMSAssist-artifact-evaluation/EMSAssist
          #command: [/bin/bash -c "tail -f /dev/null"]
          command: tail -F anything
          #network_mode: "host"
          deploy:
            resources:
              reservations:
                devices:
                  - driver: nvidia
                    capabilities: [gpu]
    volumes:
      emsassist: {}
	```

6. Clone the git repository of EMSAssist:
	```console
	$ git clone --recursive git@github.com:LENSS/EMSAssist.git`
	$ cd EMSAssist
	$ git clone --recursive git@github.com:tensorflow/examples.git
	```

7. Run docker-compose in silent mode from the terminal of current folder. For the next step, `data`, `model`, and `EMSAssist` directories along with the `docker-compose.yml` file need to be in the same folder (i.e., current folder):
	```
    $ docker-compose up -d
	# it will pull a docker container image and run it in local machine as "emsassist"
    ```

8. Log in to your running container's terminal (bash)
	```
    $ docker exec -it emsassist /bin/bash
    ```
Inside your container:

9. Activate conda environment and run `nvidia-smi` to make sure the GPU is working:
	```console
    $ conda activate emsassist-gpu
	$ nvidia-smi
    ```


<!-- ## B. Path preparation and running the evaluation

* Before we proceed, please make sure you successfully set up the environment or get the Docker image running with `nvidia-smi`

	```console
	$ export PYTHONPATH=$PWD/src/speech_recognition:$PWD/examples`

	$ export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`

	# begin the evaluation: 
	$ cd src
	# Check out the README file when you enter each sub-directory.


<!-- we create and activate a conda environment with tensorflow-gpu: `conda activate tf-gpu` -->

<!-- ## Basic Environment

| Software Environment  | Version |
| ------------- | ------------- |
| OS  | Ubuntu Server 22.04.1 LTS |
| NVIDIA Driver  | 525.85.12  |

| Hardware Environment  | Version |
| ------------- | ------------- |
| GPU  | 3 x NVIDIA A30   |
| CPU | 2 x Intel Xeon 4314 |

Before the artifact evaluation and use the open-sourced code/data, please make sure you have at least 1 NVIDIA GPU available with `nvidia-smi` command.

![nvidia-gpu](./nvidia-smi.png)

## Build the target Environment

| Software Environment  | Version |
| ------------- | ------------- |
| OS  | Ubuntu Server 22.04.1 LTS |
| NVIDIA Driver  | 525.85.12  |
| CUDA Version  | 10   |
| TensorFlow  | 2.9   | --> 


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
