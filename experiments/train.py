"""Training script.

Train a stable-baseline3 RL agent on a ReconArena environment.

Example
-------
In a terminal, run as:

    $ python train.py --algo <a2c | ppo | sac | td3 | ddpg>

"""
import os
import time
import argparse
import subprocess
import gym
import torch
import yaml
import numpy as np

from datetime import datetime
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from gym_marl_reconnaissance.envs.recon_arena import ActionType, ObsType, RewardChoice, AdversaryType, ReconArena
from gym_marl_reconnaissance.utils.utils import sync, str2bool


def main():
    """Main function.

    """
    # Set-up parser and parse the script's input arguments.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--algo',
                        default='a2c',
                        type=str,
                        choices=['a2c', 'ppo']) # Not supporting multi-discrete actions 'sac', 'td3', 'ddpg'
    parser.add_argument('--yaml', default='defaults.yaml', type=str)
    parser.add_argument('--steps', default=1e3, type=int)
    ARGS = parser.parse_args()
    # Load YAML.
    with open(os.path.dirname(os.path.abspath(__file__))+'/configurations/'+ARGS.yaml, 'r') as yaml_file:
        YAML_DICT = yaml.safe_load(yaml_file)
    # Create save path.
    experiment_name = str(ARGS.algo)+'--'+str(YAML_DICT['action_type'])+'--'+str(YAML_DICT['adv_type']) \
                        +'--n_ad-'+str(YAML_DICT['setup']['adv']) \
                        +'--n_ne-'+str(YAML_DICT['setup']['neu']) \
                        +'--n_tt-'+str(YAML_DICT['setup']['tt']) \
                        +'--n_s1-'+str(YAML_DICT['setup']['s1'])
    filename = os.path.dirname(os.path.abspath(__file__))+'/../results/exp--'+experiment_name+'--'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
    # Create the training environment.
    train_env = gym.make('recon-arena-v0', **YAML_DICT)
    check_env(train_env,
              warn=True,
              skip_render_check=True
              )
    # Define network architectures for on-policy algorithms.
    # onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
    #                        net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])])
    # Create the learning model.
    if ARGS.algo == 'a2c':
        model = A2C(a2cppoMlpPolicy,
                    train_env,
                    # policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1)
    if ARGS.algo == 'ppo':
        model = PPO(a2cppoMlpPolicy,
                    train_env,
                    # policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1)
    # Define network architectures for off-policy algorithms.
    # offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
    #                         net_arch=[512, 512, 256, 128])
    if ARGS.algo == 'sac':
        model = SAC(sacMlpPolicy,
                    train_env,
                    # policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1)
    if ARGS.algo == 'td3':
        model = TD3(td3ddpgMlpPolicy,
                    train_env,
                    # policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1)
    if ARGS.algo == 'ddpg':
        model = DDPG(td3ddpgMlpPolicy,
                    train_env,
                    # policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1)
    # Create an evaluation environment.
    eval_env = Monitor(gym.make('recon-arena-v0', **YAML_DICT), filename+'/')
    # Stopping condition on reward.
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1e5,
                                                     verbose=1
                                                     )
    # Save the best model.
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(2000),
                                 deterministic=True,
                                 render=False
                                 )
    # Dump YAML.
    YAML_DICT['algo'] = ARGS.algo
    with open(filename+'/save.yaml', 'w') as outfile:
        yaml.dump(YAML_DICT, outfile, default_flow_style=False)
    # Train.
    model.learn(total_timesteps=ARGS.steps, # 1e6,
                callback=eval_callback,
                log_interval=100
                )
    # Save the final model.
    model.save(filename+'/success_model.zip')
    print(filename)


if __name__ == '__main__':
    main()
