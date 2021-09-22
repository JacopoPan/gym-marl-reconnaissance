"""Greedy (stateless) task assignment and tracking functions.

"""
import os
import gym
import numpy as np


def greedy_task_assignment(obs,
                           action_space_shape
                           ):
    """Greedy task assignment.

    Parameters
    ----------
    obs : ndarray
        The observation returned by a call to env.step()
    act_space_shape : gym.spaces
        The shape of the action space of the environment.

    Returns
    -------
    ndarray
        A greedy task assignment action.

    """
    agents = []
    adversaries = []
    values_per_robot_entry = 5
    for i in range(int(obs.shape[0]/values_per_robot_entry)):
        this_robot = obs[5*i:5*(i+1)]
        if this_robot[1] == 1 or this_robot[1] == 2:
            agents = np.hstack([agents, this_robot])
        elif this_robot[1] == 3:
            adversaries = np.hstack([adversaries, this_robot])
    agents = agents.reshape((int(agents.shape[0]/values_per_robot_entry), values_per_robot_entry))
    adversaries = adversaries.reshape((int(adversaries.shape[0]/values_per_robot_entry), values_per_robot_entry))
    task_assignment = np.zeros(action_space_shape[0])
    for i in range(agents.shape[0]):
        own_pos = agents[i, 2:5]
        adv_pos = adversaries[:, 2:5]
        distances = np.sum((adv_pos - own_pos)**2, axis=1)
        closest_row = np.argmin(distances)
        closes_adv_id = int(adversaries[closest_row, 0])
        task_assignment[i] = closes_adv_id
        adversaries = np.delete(adversaries, obj=closest_row, axis=0)
        if adversaries.size == 0:
            break
    task_assignment = task_assignment - np.max(agents[:, 0]) - 1
    return task_assignment.astype(int)


def greedy_tracking(obs,
                    action_space_shape
                    ):
    """Greedy tracking.

    Parameters
    ----------
    obs : ndarray
        The observation returned by a call to env.step()
    act_space_shape : gym.spaces
        The shape of the action space of the environment.

    Returns
    -------
    ndarray
        A greedy tracking action.

    """
    raise RuntimeError('[ERROR] Not yet implemented.')
    # call greedy task assignment
    # loop over assignments
        # choose up, down, left, right direction from the distance vector
    # if within obstacle range AND choosen action approaches obstacles
        # take normal action instead
    # (explore if not within visibility range)
    return np.zeros(action_space_shape[0]).astype(int)
