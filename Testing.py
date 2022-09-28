from CPSsenv import CPSsenv
from Algorithms import CPSsalgorithms
import networkx as nx
from numpy.random import random, choice, seed
import matplotlib.pyplot as plt
import numpy as np
from time import time
from copy import copy, deepcopy

'''
ENVIRONMENT PARAMETERS
'''
network = 'SCADA'
T_max = 6
termination = 'one goal'
nw_params = {'Rewards': ['g5']}
rd_seed = 0
init_type = 'active list'
init_params = ['r1', 'k1', 'k4', 's1', 's3']

'''
Creating environment object
'''
env = CPSsenv(network = network, T_max = T_max, termination = termination, nw_params = nw_params)


'''
Policy object
'''
solver = CPSsalgorithms(env, init_params = init_params)

'''


# Q-Learning parameters

replicas = 5
episodes = 20000
Episodes = range(episodes) 

alpha = 0.05                                     # How fast does the agent learn
gamma = 0.99                                    # How important are future actions

epsilon = 0.7                                   # Rate at which random actions will be 
start_e_decaying = 0                            # First episode at which decay epsilon
end_e_decaying = round(episodes * 0.7)          # Last episode at which decay epsilona
epsilon_decay_value = (epsilon - 0.03) / (end_e_decaying - start_e_decaying)


# Q-Learning Training

start = time()
q_table = {} 
episodes_rewards, successes, num_states = [], [], []
alphas = {attack:0 for attack in env.Attack_steps} 

for episode in Episodes:
    
    # Initializing environment
    
    state, available_actions = env.reset(init_type = init_type, init_params = init_params, rd_seed = rd_seed + episode) 
    done = False
    episode_reward = 0
    chosen_actions = []

    while not done:

        old_state = deepcopy(state)
        action = solver.Q_Learning_action(q_table, state, available_actions, epsilon = epsilon)

        chosen_actions.append(action)
        new_state, available_actions, reward, done, _ = env.step(action)
        episode_reward += reward

        q_table = solver.Q_Learning_update(q_table, old_state, action, reward, new_state, alpha = alpha, gamma = gamma)

        state = new_state

    if end_e_decaying >= episode >= start_e_decaying:       # Decay epsilon
        epsilon -= epsilon_decay_value

    episodes_rewards.append(episode_reward)
    successes.append(int(_['Success']))
    num_states.append(len(list(q_table.keys())))

    if _['Success']:
        for action in set(chosen_actions):
            alphas[action] += 1
                                               


# Training stats
       
print('\n############## Training done ##############\n')
print(f'Training time:                 {round(time() - start,2)} s')
print(f'Success prob on last 10% ep:   {round(sum(successes[int(0.9*episodes):])/(0.1*episodes),2)}')


# Mobiled-averaged rewards for plotting 

avg_episodes = 250
average_rewards = []
average_probs = []
for episode in Episodes:
    if episode <= avg_episodes/2:
        av_rw = sum(episodes_rewards[:avg_episodes])
        av_num = sum(successes[:avg_episodes]) 
        average_rewards.append(av_rw/(avg_episodes))
        average_probs.append(av_num/((avg_episodes)))
    elif episode <= episodes - avg_episodes/2:
        lower = int(episode-avg_episodes/2)
        upper = int(episode+avg_episodes/2)
        av_rw = sum(episodes_rewards[lower:upper])
        av_num = sum(successes[lower:upper])
        average_rewards.append(av_rw/(avg_episodes))
        average_probs.append(av_num/(avg_episodes))
    else:
        av_rw = sum(episodes_rewards[episodes - avg_episodes:])
        av_num = sum(successes[episodes - avg_episodes:])  
        average_rewards.append(av_rw/(avg_episodes))
        average_probs.append(av_num/((avg_episodes)))

plots = True

if plots:
    plt.plot(average_rewards, color = 'purple')
    plt.title('Average reward through the episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Average reward')
    plt.show()

    
    # Ploting succesfull episodes
    
    plt.plot(average_probs, color = 'blue')
    plt.title('Succesfull attacks through the episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Proportion of succesfull episodes')
    plt.show()


    
    # Ploting number of states
    
    plt.plot(num_states, color = 'pink')
    plt.title('Number of explored states')
    plt.xlabel('Episodes')
    plt.ylabel('Explored states')
    plt.show()

    
    # Ploting frequency of actions on succesfull attacks
    
    plt.bar(list(alphas.keys()), list(alphas.values()), color = 'orange')
    plt.title('Recurence of attacks through successfull episodes')
    plt.xlabel('Attacks')
    plt.ylabel('Frequency per attack')
    plt.show()



# Q-Learning testing

episodes_rewards = [] 
sss = []
printt = False
for episode in range(50):
    state, available_actions = env.reset(init_type = init_type, init_params = init_params, rd_seed = rd_seed * 2)
    done = False
    episode_reward = 0
    summary = []

    while not done:

        old_state = deepcopy(state)
        action = solver.Q_Learning_action(q_table, state, available_actions, epsilon = 0)

        new_state, available_actions, reward, done, _ = env.step(action)
        episode_reward += reward

        summary.append((old_state[0], old_state[3], action))

        state = new_state

    if _['Success'] and not printt:
        for i in summary:
            print(i)
        printt = True
    #     print('Successful testing episode')
    # else:
    #     print('Unsuccessful testing episode')
    sss.append(int(_['Success']))
    episodes_rewards.append(episode_reward)


# Testing stats
 
print('\n############## Testing done ##############\n')
print(f'Success rate: {round(sum(sss)/50,2)}')
print(f'Avg reward:   {round(sum(episodes_rewards)/50,2)}')

state = [{'r1': 1, 'r2': 0, 'r3': 0, 'r4': 0, 'r5': 0, 'r6': 0, 'r7': 0, 'r8': 0}, {'k1': 1, 'k2': 0, 'k3': 0, 'k4': 1}, {'s1': 1, 's2': 0, 's3': 1}, {'g1': 0, 'g2': 0, 'g3': 0, 'g4': 0, 'g5': 0}]
print(q_table[tuple(solver.translate_state(state))])
'''


'''
Value iteration
'''
gamma = 0.9
theta = 0.00009
state, available_actions = env.reset(init_type = init_type, init_params = init_params)
States = solver.generate_states(env, loops = 50000, load = False)


V_hat, policy = solver.Value_Iteration(env, States, gamma, theta)
for state in policy.keys():
    print(f'On state {solver.ret_state(state)} -> {policy[state]}')


