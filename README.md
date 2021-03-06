# PILCO

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An Implementation of PILCO: Probabilistic Inference for Learning Control on the Pendulum
environment

<p align="center">
  <img src="pendulum.png" width="500" title="logo">
</p>

## Install

We recommend using a conda virtual environment. Installation assumes that the
user already has a MuJuCo licence available at: https://www.roboti.us/license.html

```
conda create -n pilco python==3.7
conda activate pilco

pip install tensorflow==2.2.0
pip install gpflow==2.0.0
pip install gast==0.3.3
pip install gym
pip3 install -U 'mujoco-py<2.1,>=2.0'
conda install matplotlib pandas pytest
```

## Usage

The example may be run as follows:

```
python pendulum_example.py
```

where the default hyperparameter values have been configured to produce a plot resembling the
one below. Random seeds are not fixed and so different plots will be obtained every time
the script is run. There is considerable scope for hyperparameter tuning!

<p align="center">
  <img src="plots/pendulum_returns.png" width="500" title="logo">
</p>

## References

The following excellent open-source projects: https://github.com/nrontsis and 
https://github.com/aidanscannell/pilco-tensorflow proved to be highly valuable references.
 