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
        self.env = env

    def Q_Learning_action(self, q_table, state, available_actions, epsilon = 0.7):
            
        sttate = self.translate_state(state)
        # print('State: ', state)
        if random() < epsilon or tuple(sttate) not in list(q_table.keys()):
            action = choice(available_actions)
        elif {i:j for i,j in q_table[tuple(sttate)].items() if j in available_actions} == {}:
            action = choice(available_actions)
        else:
            real_dict = {i:j for i,j in q_table[tuple(sttate)].items() if j in available_actions}
            action = max(real_dict, key = real_dict.get)

        return action


    def Q_learning_update(self, q_table, state, action, reward, new_state, alpha = 0.1, gamma = 0.95):
        sttate = self.translate_state(state)
        new_sttate = self.translate_state(new_state)

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
            
        return q_table


    def translate_state(self, state):
        trans_state = []
        for j in range(4):
            for i in state[j].values():
                trans_state.append(i)
        return trans_state
        