# PILCO
An Implementation of PILCO: Probabilistic Inference for Learning Control

## Install

We recommend using a conda virtual environment. Installation assumes that the
user already has a MuJuCo licence available at: https://www.roboti.us/license.html

```
conda create -n pilco python==3.7
conda activate pilco

pip install tensorflow==2.2.0
pip install gpflow==2.0.0
pip install gym
pip3 install -U 'mujoco-py<2.1,>=2.0'
conda install matplotlib pandas pytest
```

## Usage