# gym-marl-reconnaissance

Gym environments for heterogeneous multi-agent reinforcement learning in non-stationary worlds

> This repository's `master` branch is actively developed, please `git pull` frequently and feel free to open new [issues](https://github.com/JacopoPan/gym-marl-reconnaissance/issues) for any undesired, unexpected, or (presumably) incorrect behavior. Thanks 🙏

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

## Use

Step an environment with random action inputs
```
$ python ./experiments/debug.py --gui True --record False --debug True
```
Learn using [`stable-baselines3`](https://stable-baselines3.readthedocs.io/en/master/)
```
$ python ./experiments/train.py --algo <a2c | ppo | sac | td3 | ddpg>
```
Replay the trained agent
```
$ python ./experiments/test.py --exp ./results/exp-<algo>-<date>_<time>
```

<img src="figures/task.gif" alt="figure" width="390"> <img src="figures/track.gif" alt="figure" width="390">

## References
TBD

-----
> University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute](https://github.com/VectorInstitute) /  [Mitacs](https://www.mitacs.ca/en/projects/multi-agent-reinforcement-learning-decentralized-uavugv-cooperative-exploration)
