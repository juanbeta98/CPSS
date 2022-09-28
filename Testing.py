from CPSsenv import CPSsenv
from Algorithms import CPSsalgorithms
import networkx as nx
from numpy.random import random, choice, seed
import matplotlib.pyplot as plt
import numpy as np

'''
ENVIRONMENT PARAMETERS
'''
network = 'SCADA'
T_max = 5
termination = 'one goal'
nw_params = {'Rewards': ['g1']}
rd_seed = 0
init_type = 'active list'
init_params = ['r1']

'''
Creating environment object
'''
env = CPSsenv(network = network, T_max = T_max, termination = termination, nw_params = nw_params)


'''
Policy object
'''
solver = CPSsalgorithms(env)


'''
Q-Learning parameters
'''
replicas = 10
episodes = 20000
Episodes = range(episodes) 

alpha = 0.1                     # How fast does the agent learn
gamma = 0.95                    # How important are future actions

epsilon = 0.7                    # Rate at which random actions will be 
start_e_decaying = 1             # First episode at which decay epsilon
end_e_decaying = episodes // 6    # Last episode at which decay epsilona
epsilon_decay_value = epsilon / (end_e_decaying - start_e_decaying)

'''
Q-Learning Training
'''
q_table = {} 
episodes_rewards, succeses, num_states = [], [], []

for episode in Episodes:
    '''
    Initializing environment
    '''
    state, available_actions = env.reset(init_type = init_type, init_params = {}) 
    done = False
    episode_reward = 0

    while not done:

        action = solver.Q_Learning_action(q_table, state, available_actions, epsilon = epsilon)

        new_state, available_actions, reward, done, _ = env.step(action)
        episode_reward += reward

        q_table = solver.Q_Learning_update(q_table, state, action, reward, new_state, alpha = alpha, gamma = gamma)

    if end_e_decaying >= episode >= start_e_decaying:       # Decay epsilon
        epsilon -= epsilon_decay_value

    episodes_rewards.append(episode_reward)
    succeses.append(_['Success'])
    num_states.append(len(list(q_table.keys())))
                                               
        


plt.plot(episodes_rewards, color = 'purple')
plt.title('Average reward through the episodes')
plt.xlabel('Episodes')
plt.ylabel('Average reward')
plt.show()