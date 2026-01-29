# [ICLR 2026] Policy Likelihood-based Query Sampling and Critic-Exploited Reset for Efficient Preference-based Reinforcement Learning

## Description
This is official implementation of [PoLiCER](https://openreview.net/forum?id=ITeuGb2bYg&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)) for pixel-based state inputs. Our implementation is based on the official codebase of [B-Pref](https://github.com/rll-research/BPref) and [SURF](https://github.com/alinlab/SURF/tree/pixel). For meta-world environment wrapper, we also benefit from [DrM](https://github.com/XuGW-Kevin/DrM?tab=readme-ov-file). We appreciate their insightful works. 

> **Policy Likelihood-based Query Sampling and Critic-Exploited Reset for Efficient Preference-based Reinforcement Learning**<br>
> Jongkook Heo, Jaehoon Kim, Young Jae Lee, Min Gu Kwak, Youngjoon Park, Seoung Bum Kim<br>
> 
>**Abstract**: <br>
Preference-based reinforcement learning (PbRL) enables agent training without explicit reward design by leveraging human feedback. Although various query sampling strategies have been proposed to improve feedback efficiency, many fail to enhance performance because they select queries from outdated experiences with low likelihood under the current policy. Such queries may no longer represent the agent's evolving behavior patterns, reducing the informativeness of human feedback. To address this issue, we propose a policy likelihood-based query sampling and critic-exploited reset (PoLiCER). Our approach uses policy likelihood-based query sampling to ensure that queries remain aligned with the agent’s evolving behavior. However, relying solely on policy-aligned sampling can result in overly localized guidance, leading to overestimation bias, as the model tends to overfit to early feedback experiences. To mitigate this, PoLiCER incorporates a dynamic resetting mechanism that selectively resets the reward estimator and its associated Q-function based on critic outputs. Experimental evaluation across diverse locomotion and robotic manipulation tasks demonstrates that PoLiCER consistently outperforms existing PbRL methods. Our code is available at https://github.com/JongKook-Heo/PoLiCER.

## How to Install
Same with vector-based control branch

### Docker Setting and Install Mujoco 2.1.0

```bash
# run docker container
docker run -it -d --shm-size=64g --gpus=all -v /your/drive/location:/mnt/hdd/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --env QT_X11_NO_MITSHM=1 --name PoLiCER_image pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
apt-get update
apt-get install sudo
sudo apt update
cd ../mnt/hdd/workspace
apt-get install wget
apt-get install nano

# install mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir /root/.mujoco
tar -xvzf mujoco210-linux-x86_64.tar.gz -C /root/.mujoco
sudo apt install libglew-dev libgl-dev -y

# copy and paste following commands into /root/.bashrc
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
export PATH="$LD_LIBRARY_PATH:$PATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so:/usr/lib/x86_64-linux-gnu/libGLEW.so
export DISPLAY=:1

sudo apt-get install x11-apps

conda install -c conda-forge xorg-libx11 xorg-libxcomposite xorg-libxcursor xorg-libXdamage xorg-libXext xorg-libXfixes xorg-libXi xorg-libXinerama xorg-libXrandr -y

cd root
source .bashrc

sudo apt-get install python3-dev build-essential libssl-dev libffi-dev libxml2-dev -y
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libegl1 libopengl0 -y


sudo apt-get install libxslt1-dev zlib1g-dev python3-pip
conda install libffi==3.3
conda install git
git clone https://github.com/openai/mujoco-py
cd mujoco-py
pip install -r requirements.txt
pip install -r requirements.dev.txt

pip3 install -e . --no-cache

sudo apt install -y mesa-utils
```

### Install Dependencies
Python 3.8.12

```bash
pip install mujoco==2.3.5
pip install gym==0.25.2
pip install dm_control==1.0.12
pip install git+https://github.com/denisyarats/dmc2gym.git
pip install tensorboard termcolor pybullet scikit-image
pip install hydra-core==1.0.4
pip install "cython<3"

pip install opencv-python
pip install imageio imageio[ffmpeg]
pip install einops
```

### Metaworld Dependency
The Metaworld package is still under development and continually updated. Therefore, to replicate our experiments, you need to install Metaworld using the following commands:

```bash
wget https://github.com/Farama-Foundation/Metaworld/archive/refs/tags/v2.0.0.tar.gz
tar -xvzf v2.0.0.tar.gz
cd Metaworld-2.0.0
pip install metaworld -e .
```

### Hydra Dependency
We used hydra with current version of 1.0.4, while original [B-Pref](https://github.com/rll-research/BPref) used version 0.x.
We slightly modified hydra configuration in **config** folder and *hydra.main()* args in all **train_x.py**. It does not affect the experiment, but only affect compatibility for hydra version.
For more details, please refer to [hydra config path changes](https://hydra.cc/docs/upgrades/0.11_to_1.0/config_path_changes/)

## How to run
### Synthetic annotator experiment with image-based observation (DMC)
```bash
export device=0
bash scripts/dmc.sh
```

### Synthetic annotator experiment with image-based observation (Meta-world)
```bash
unset LD_PRELOAD
export LD_PRELOAD=""
export device=0
bash scripts/meta.sh
```

## Citation (To be specified after camera ready)
```latex
@inproceedings{
anonymous2026policy,
title={Policy Likelihood-based Query Sampling and Critic-Exploited Reset for Efficient Preference-based Reinforcement Learning},
author={Anonymous},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=ITeuGb2bYg}
}
```