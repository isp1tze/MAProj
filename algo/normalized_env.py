'''
Implemented by ghliu
https://github.com/ghliu/pytorch-ddpg/blob/master/normalized_env.py
'''

import gym
import numpy as np

# https://github.com/openai/gym/blob/master/gym/core.py
class ActionNormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    def __init__(self, env):
        super(ActionNormalizedEnv, self).__init__(env=env)
        self.action_high = 1.
        self.action_low = -1.

    def action(self, action):
        act_k = (self.action_high - self.action_low)/ 2.
        act_b = (self.action_high + self.action_low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_high - self.action_low)
        act_b = (self.action_high + self.action_low)/ 2.
        return act_k_inv * (action - act_b)

class ObsEnv(gym.ObservationWrapper):
    """ Wrap action """
    def __init__(self, env):
        super(ObsEnv, self).__init__(env=env)
        self.action_high = 1.
        self.action_low = -1.

    def observation(self, observation):
        '''
        :param observation:
        :return: removal of agent.state.c (bool : communication)
        '''
        return [obs[:14] for obs in observation]



def reward_from_state(n_state):
    rew = []

    for state in n_state:

        obs_landmark = np.array(state[4:10])
        agent_reward = 0
        for i in range(3):

            sub_obs = obs_landmark[i*2: i*2+2]
            dist = np.sqrt(sub_obs[0]**2 + sub_obs[1]**2)

            # if dist < 0.4: agent_reward += 0.3
            if dist < 0.2: agent_reward += 0.5
            if dist < 0.1: agent_reward += 1.


        otherA = np.array(state[10:12])
        otherB = np.array(state[12:14])
        dist = np.sqrt(otherA[0] ** 2 + otherA[1] ** 2)
        if dist < 3.1:  agent_reward -= 0.25
        dist = np.sqrt(otherB[0] ** 2 + otherB[1] ** 2)
        if dist < 3.1:  agent_reward -= 0.25

        rew.append(agent_reward)

    return rew

'''
def reward_from_state(state):
    rew = 0

    _state = state[0]
    dist = np.sqrt(_state[2]**2 + _state[3]**2)

    if dist < 0.4: rew += 2
    elif dist < 0.2: rew += 4
    elif dist < 0.1: rew += 5

    return np.array([rew], dtype=np.float32)
'''