

from CPSsenv import CPSsenv
import networkx as nx
from numpy.random import random, choice, seed
import matplotlib.pyplot as plt
import numpy as np

'''
ENVIRONMENT PARAMETERS
'''
network = 'SCADA'
T_max = 5
termination = 'One goal'
params = {}
rd_seed = 0

'''
Creating environment object
'''
env = CPSsenv(network = network, T_max = T_max, termination = termination, params = params)

'''
Initializing environment
'''
initial_state = env.initial_state
initial_state[0]['r1'] = True
initial_state[0]['r6'] = True
initial_state[2]['s1'] = True

state, available_actions = env.reset(init_state = initial_state, rd_seed = rd_seed)
print(env.available_actions)