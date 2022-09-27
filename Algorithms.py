'''
Authors: 
- Juan Betancourt 
- Germán Pardo
'''
from CPSsenv import CPSsenv
import networkx as nx
from numpy.random import random, choice, seed
import matplotlib.pyplot as plt
import numpy as np


class CPSsalgorithms():

    def __init__(self, env):
        pass

    def Q_Learning(self, env, init_state, episodes, alpha = 0.1, gamma = 0.95, epsilon = 0.7,
                   start_e_decaying = 1, end_e_decaying = 3):
        
        Episodes = range(episodes)
        q_table = {}

        episodes_rewards, succeses, num_states = [], [], []

        end_e_decaying = episodes // end_e_decaying    # Last episode at which decay epsilon
        epsilon_decay_value = epsilon / (end_e_decaying - start_e_decaying)  
    
        for episode in Episodes:
            
            episode_reward = 0
            state, available_actions = env.reset(init_state)
            sttate = self.translate(state)
            done = False

            while not done:

                # print('State: ', state)
                if random() < epsilon or tuple(sttate) not in list(q_table.keys()):
                    action = choice(available_actions)
                elif {i:j for i,j in q_table[tuple(sttate)].items() if j in available_actions} == {}:
                    action = choice(available_actions)
                else:
                    real_dict = {i:j for i,j in q_table[tuple(sttate)].items() if j in available_actions}
                    action = max(real_dict, key = real_dict.get)

                # print(action)
                new_state, available_actions, reward, done, _ = env.step(action)
                episode_reward += reward
                new_sttate = translate_state(new_state)

                if tuple(sttate) not in list(q_table.keys()):
                            
                    q_table[tuple(sttate)] = {}
                    q_table[tuple(sttate)][action] = reward
                
                elif new_sttate not in list(q_table.keys()) or action not in q_table[tuple(sttate)].keys():
                            
                    q_table[tuple(sttate)][action] = reward
                    
                else:
                    
                    max_future_q = max(list(q_table[new_sttate].values()))    # Minimum value of the arriving state
                    current_q = q_table[tuple(sttate)][action]                   # Value of current state and action
                
                    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)               
                
                    q_table[tuple(sttate)][action] = new_q     # Update Q Value for current state and action
                    
                state = new_state
                sttate = new_sttate
        
        if end_e_decaying >= episode >= start_e_decaying:       # Decay epsilon
                epsilon -= epsilon_decay_value
            
        episodes_rewards.append(episode_reward)
        succeses.append(_['Success'])


        def translate_state(self, state):
            trans_state = []
            for j in range(4):
                for i in state[j].values():
                    trans_state.append(i)
            return trans_state
        