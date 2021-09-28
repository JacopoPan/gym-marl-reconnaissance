# gym-marl-reconnaissance

Gym environments for heterogeneous multi-agent reinforcement learning in non-stationary worlds

> This repository's `master` branch is work in progress, please `git pull` frequently and feel free to open new [issues](https://github.com/JacopoPan/gym-marl-reconnaissance/issues) for any undesired, unexpected, or (presumably) incorrect behavior. Thanks 🙏

## Install on Ubuntu/macOS

(optional) Create and access a Python 3.7 environment using [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
```
$ conda create -n recon python=3.7                                 # Create environment (named 'recon' here)
$ conda activate recon                                             # Activate environment 'recon'
```
Clone and install the `gym-marl-reconnaissance` repository 
```
$ git clone https://github.com/JacopoPan/gym-marl-reconnaissance   # Clone repository
$ cd gym-marl-reconnaissance                                       # Enter the repository
$ pip install -e .                                                 # Install the repository
```

## Configure

Set the parameters of the simulation environment
```
seed: -1
ctrl_freq: 2
pyb_freq: 30
gui: False
record: False
episode_length_sec: 30
action_type: 'task_assignment'      # Alternatively, 'tracking'
obs_type: 'global'
reward_choice: 'reward_c'
adv_type: 'avoidant'                # Alternatively, 'blind'
visibility_threshold: 12
setup:
  edge: 10
  obstacles: 0
  tt: 1
  s1: 1
  adv: 2
  neu: 1
debug: False
```

## Use

Step an environment with random action inputs
```
$ python3 ./experiments/debug.py --random True
```
Step an environment with a greedy policy (**only for `task_assignment`**)
```
$ python3 ./experiments/debug.py
```
Learn using [`stable-baselines3`](https://stable-baselines3.readthedocs.io/en/master/)
```
$ python ./experiments/train.py --algo <a2c | ppo> --yaml <filname in ./experiments/configurations/>
```
Replay a trained agent
```
$ python ./experiments/test.py --exp ./results/exp--<algo>--<config>--<date>_<time>
```

<img src="figures/task.gif" alt="figure" width="400"> <img src="figures/track.gif" alt="figure" width="400">

-----
> University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute](https://github.com/VectorInstitute) /  [Mitacs](https://www.mitacs.ca/en/projects/multi-agent-reinforcement-learning-decentralized-uavugv-cooperative-exploration)
