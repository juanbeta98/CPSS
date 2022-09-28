'''
Authors: 
- Juan Betancourt 
- Germán Pardo
'''
from zoneinfo import available_timezones
from CPSsenv import CPSsenv
import networkx as nx
from numpy.random import random, choice, seed
import matplotlib.pyplot as plt
import numpy as np
import ast
from copy import deepcopy


class CPSsalgorithms():

    def __init__(self, init_type, init_params):
        self.init_type = init_type
        self.init_params = init_params

    def Q_Learning_action(self, q_table, state, available_actions, epsilon = 0.7):
            
        sttate = self.translate_state(state)
        # print('State: ', state)
        if random() < epsilon or tuple(sttate) not in list(q_table.keys()):
            action = choice(available_actions)
        elif {i:j for i,j in list(q_table[tuple(sttate)].items()) if i in available_actions} == {}:
            action = choice(available_actions)
        else:
            real_dict = {i:j for i,j in list(q_table[tuple(sttate)].items()) if i in available_actions}
            action = max(real_dict, key = real_dict.get)

        return action


    def Q_Learning_update(self, q_table, state, action, reward, new_state, alpha = 0.1, gamma = 0.95):
        sttate = self.translate_state(state)
        new_sttate = self.translate_state(new_state)

        if tuple(sttate) not in list(q_table.keys()):
                    
            q_table[tuple(sttate)] = {}
            q_table[tuple(sttate)][action] = reward
        
        elif tuple(new_sttate) not in list(q_table.keys()) or action not in list(q_table[tuple(sttate)].keys()):
                    
            q_table[tuple(sttate)][action] = reward
            
        else:
            
            max_future_q = max(list(q_table[tuple(new_sttate)].values()))    # Minimum value of the arriving state
            current_q = q_table[tuple(sttate)][action]                   # Value of current state and action
        
            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)               
        
            q_table[tuple(sttate)][action] = new_q     # Update Q Value for current state and action
            
        return q_table


    def Value_Iteration(self, env, States, gamma, theta):
        v_hat = {}
        for ii in States:
            v_hat[tuple(ii)] = 0
        # v_hat = {tuple(ii):0 for ii in States}

        print(v_hat)
        policy = {tuple(state):'' for state in States}

        while True:

            delta = 0

            for index in range(len(States)):
 
                state = deepcopy(self.ret_state(States[index]))
                sttate = States[index]

                st, available_actions = env.reset(init_type = 'env state', init_params = state) 

                old_val = deepcopy(v_hat[tuple(sttate)])

                mejor_accion = ''
                mejor_valor = -2
                valor = 0

                if available_actions != []:
                    for action in available_actions:

                        st, av_act = env.reset(init_type = 'env state', init_params = state) 
                        s_prima, av_act, reward, done, _ = env.step(action, stochastic = False)
                        s_primaa = self.translate_state(s_prima)
                    
                        if s_primaa in States and not done:
                            valor = env.nw.nodes()[action]['p'] * ((reward) + gamma * v_hat[tuple(s_primaa)]) + \
                                (1-env.nw.nodes()[action]['p']) * (-env.nw.nodes()[action]['C'] + gamma * v_hat[tuple(s_primaa)])

                        elif s_primaa in States and done:
                            valor = env.nw.nodes()[action]['p'] * (reward) + \
                                (1-env.nw.nodes()[action]['p']) * env.nw.nodes()[action]['C']

                        else:
                            States.append(s_primaa)
                            valor = 0
                            v_hat[tuple(s_primaa)] = valor

                        if valor > mejor_valor:
                            mejor_valor = valor
                            mejor_accion = action

        
                v_hat[tuple(sttate)] = mejor_valor
                policy[tuple(sttate)] = mejor_accion

                delta =  max(delta, abs(v_hat[tuple(sttate)]- old_val))

            if delta < theta:
                break

        return v_hat, policy



    def generate_states(self, env, loops = 50000, load = False):
  
        states = []
        if load == False:
            for iter in range(loops):
                done = False
                state, available_actions = env.reset(init_type = self.init_type, init_params = self.init_params)  
                if iter == 0:   states.append(self.translate_state(state)) 

                while not done:
                    action = choice(available_actions)
                    st_prime, available_actions, reward, done, _ = env.step(action, stochastic = False)
                        
                    if not done and st_prime not in states:
                        states.append(self.translate_state(st_prime))
        
        else:
            inter_list = []
            with open(r'statesFull.txt', 'r') as fp:
                for line in fp:
                    # remove linebreak from a current name
                    # linebreak is the last character of each line
                    x = line[:-1]
                    inter_list.append(x)
                    states.append(ast.literal_eval(x))

        return states

    def translate_state(self, state):
        trans_state = []
        for j in range(4):
            for i in state[j].values():
                trans_state.append(i)
        return trans_state
    
    # TODO GENERALIZE FOR DIFFERENT NETWORKS
    def ret_state(self, state):
        dic1 = {};  dic2 = {};  dic3 = {};   dic4 = {}
        ii = 0
        for i in range(1,9):
            dic1['r'+str(i)] = state[ii]
            ii += 1
        for i in range(1,5):
            dic2['k'+str(i)] = state[ii]
            ii += 1
        for i in range(1,4):
            dic3['s'+str(i)] = state[ii]
            ii += 1
        for i in range(1,6):
            dic4['g'+str(i)] = state[ii]
        
        return [dic1, dic2, dic3, dic4]
        