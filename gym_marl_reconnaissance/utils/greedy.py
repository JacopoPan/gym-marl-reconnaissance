"""Greedy (stateless) task assignment and tracking functions.

"""
import os
import gym
import numpy as np


def greedy_task_assignment(obs,
                           obs_space,
                           action_space
                           ):
    """Greedy task assignment.

    Parameters
    ----------
    obs : ndarray
        The observation returned by a call to env.step()
    obs_space : gym.spaces
        The observation space of the environment.
    act_space : gym.spaces
        The action space of the environment.

    Returns
    -------
    ndarray
        A greedy task assignment action.

    """
    print('tbd')
    return np.zeros(3)


def greedy_tracking(obs,
                    obs_space,
                    action_space
                    ):
    """Greedy tracking.

    Parameters
    ----------
    obs : ndarray
        The observation returned by a call to env.step()
    obs_space : gym.spaces
        The observation space of the environment.
    act_space : gym.spaces
        The action space of the environment.

    Returns
    -------
    ndarray
        A greedy tracking action.

    """
    print('tbd')
    return np.zeros(3)
