from CPSsenv import CPSsenv
import networkx as nx
from numpy.random import random, choice
import matplotlib.pyplot as plt
import numpy as np
import ast

'''
Translation function for the state
'''
def gen_state(state):
    s_prime = []
    for value in state[0].values():
        s_prime.append(value)
    for value in state[1].values():
        s_prime.append(value)
    for value in state[2].values():
        s_prime.append(value)
    for value in state[3].values():
        s_prime.append(value)
    
    return s_prime

def ret_state(state):
    dic1 = {}
    dic2 = {}
    dic3 = {}
    dic4 = {}
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


'''
Creating networkx graph of the problem
'''
network = nx.DiGraph()

nodes = [('r1', {'type': 'Access', 'D': True}), ('r2', {'type': 'Access', 'D': False}), 
         ('r3', {'type': 'Access', 'D': False}), ('r4', {'type': 'Access', 'D': False}),
         ('r5', {'type': 'Access', 'D': False}),('r6', {'type': 'Access', 'D': False}),
         ('r7', {'type': 'Access', 'D': False}),('r8', {'type': 'Access', 'D': False}),
         
         ('k1', {'type': 'Knowledge', 'D': False}),
         ('k2', {'type': 'Knowledge', 'D': False}),
         ('k3', {'type': 'Knowledge', 'D': False}),
         ('k4', {'type': 'Knowledge', 'D': False}),
         
         ('s1', {'type': 'Skill', 'D': False}), ('s2', {'type': 'Skill', 'D': False}),
         ('s3', {'type': 'Skill', 'D': False}),
         
         ('g1', {'type': 'Goal', 'D': False, 'R': 1}), ('g2', {'type': 'Goal', 'D': False, 'R': 1}),
         ('g3', {'type': 'Goal', 'D': False, 'R': 1}), ('g4', {'type': 'Goal', 'D': False, 'R': 1}),
         ('g5', {'type': 'Goal', 'D': False, 'R': 1}),
         
         ('a1', {'type': 'Attack step', 'D': False, 'p': 1, 'C': 1}), ('a2', {'type': 'Attack step', 'D': False, 'p': 0.3, 'C': 60}),
         ('a3', {'type': 'Attack step', 'D': False, 'p': 0.5, 'C': 60}), ('a4', {'type': 'Attack step', 'D': False, 'p': 0.1, 'C': 120}),
         ('a5', {'type': 'Attack step', 'D': False, 'p': 0.3, 'C': 90}), ('a6', {'type': 'Attack step', 'D': False, 'p': 0.6, 'C': 30}),
         ('a7', {'type': 'Attack step', 'D': False, 'p': 0.75, 'C': 70}), ('a8', {'type': 'Attack step', 'D': False, 'p': 0.8, 'C': 20}),
         ('a9', {'type': 'Attack step', 'D': False, 'p': 0.95, 'C': 10}), ('a10', {'type': 'Attack step', 'D': False, 'p': 0.5, 'C': 90}),
         ('a11', {'type': 'Attack step', 'D': False, 'p': 0.85, 'C': 20}),
         
         ('a12', {'type': 'Attack step', 'D': False, 'p': 1, 'C': 1}),('a13', {'type': 'Attack step', 'D': False, 'p': 0.7, 'C': 10}),
         ('a14', {'type': 'Attack step', 'D': False, 'p': 1, 'C': 1}),('a15', {'type': 'Attack step', 'D': False, 'p': 1, 'C': 1}),
         ('a16', {'type': 'Attack step', 'D': False, 'p': 0.65, 'C': 90}),('a17', {'type': 'Attack step', 'D': False, 'p': 0.75, 'C': 150}),
         ('a18', {'type': 'Attack step', 'D': False, 'p': 0.5, 'C': 20})]
network.add_nodes_from(nodes)
edges = [('r1','a1'), ('r1','a2'), ('r1','a3'), ('r1','a4'), ('r1','a5'),
         ('k1','a1'), 
         ('k2','a12'), 
         ('k3','a14'),
         ('k4','a15'),
         ('s1','a2'),
         ('s2','a3'), ('s2','a4'), ('s2','a5'),
         ('s3','a16'), ('s3','a7'),
         ('a1','r2'),
         ('a2','r2'),
         ('a3','r2'),
         ('a4','r3'),
         ('a5','r4'),
         ('r2','a4'), ('r2','a5'), ('r2','a6'), ('r2','a9'),
         ('r3','a3'), ('r3','a5'), ('r3','a8'), ('r3','a9'),
         
         ('r6','a12'), ('r6','a14'), ('r3','a13'),
         ('r7','a15'),
         ('r8','a16'), ('r8','a17'),
         ('r4','a3'), ('r4','a4'), ('r4','a9'),
         ('a6','r5'),
         ('r5','a7'), ('r5','a8'), ('r5','a9'),
         ('a7','g1'), 
         ('a8','g2'), 
         ('a9','g3'),
         ('g1','a10'), ('g1','a11'), ('g1','a18'),
         ('a10','g4'),
         ('a11','g2'),
         ('a12','r2'),
         ('a13','r7'),
         ('a14','r5'),
         ('a15','r8'),
         ('a16','g1'),
         ('a17','g5'),
         ('a18','g5'),]
network.add_edges_from(edges)
env = CPSsenv(network)
    
'''
Creating a list with a large number of feasible states
'''  
# states = []
# # !TODO Crear primer estado inicial factible para inicializar la generación de otros estados
# # Lista de los posibles estados
# for iter in range(50000):
#     if iter%5000==0: 
#         print(iter)
#     done = False
#     state, available_actions = env.reset(l = False, random_start = False)   
#     states.append(gen_state(state))

#     while not done:

#         action = choice(available_actions)
#         st_prime, available_actions, reward, done, _ = env.step2(action)

#         if st_prime not in states:
#             states.append(gen_state(st_prime))

# # open file in write mode
# with open(r'/Users/juanbeta/My Drive/Investigación/CPSS/Code/states.txt', 'w') as fp:
#     for item in states:
#         # write each item on a new line
#         fp.write("%s\n" % item)

inter_list = []
states = []
# open file and read the content in a list
with open(r'/Users/juanbeta/My Drive/Investigación/CPSS/Code/states.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]
        # inter_list.append(x)
        states.append(ast.literal_eval(x))
print(len(states))

# for i in range(len(inter_list)):
#     states.append(ast.literal_eval(inter_list[i]))


'''
Storing real transition probabilites and costs
'''
prob_real_exito = {}
c = {}
for node in nodes:
    if node[1]['type'] == 'Attack step':
        prob_real_exito[node[0]] = node[1]['p']
        c[node[0]] = -node[1]['C']

'''
Value iteration parameters
'''
gamma = 0.9
delta = 0#-1000000
v_hat = {tuple(i):0 for i in states}

'''
Value iteration
'''
print('Starting value iteration')
while True:

    error = 0
    env = CPSsenv(network)
    
    for i in range(len(states)):
        
        state = states[i]
        # Inicializar ambiente en el estado de interes   
        s, available_actions = env.reset(l = ret_state(state), random_start = False)
        
        # Guardamos el valor viejo
        old_val = v_hat[tuple(state)]
        
        # Escoger mejor accion 
        mejor_accion = 0#-10000
        mejor_valor = -100000
        
        for a in available_actions:

            info = env.reset(l = ret_state(state), random_start = False)           

            #### siempre llega al mismo s_prima
            s_prima, av_act, reward, done, _ = env.step2(a)
            s_primaa = gen_state(s_prima)
            
            if s_primaa in states:
                valor = prob_real_exito[a] * ((reward) + gamma * v_hat[tuple(s_primaa)]) + (1-prob_real_exito[a]) * (c[a] + gamma * v_hat[tuple(state)])
            else:
                states.append(s_primaa)
                v_hat[tuple(s_primaa)] = 0
            
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = a

        
        v_hat[tuple(state)] = mejor_valor
        delta =  max(delta, abs(v_hat[tuple(s_primaa)]- old_val))
        if i%20000==0:
            print('State'+str(i))

    print(delta)

    if delta < 1e-12:
        break