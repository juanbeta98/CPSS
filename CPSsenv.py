'''
Authors: 
- Juan Betancourt
- Germán Pardo
'''

import networkx as nx
from numpy.random import random, seed

class CPSsenv():

    def __init__(self, network) -> None:

        self.nw = network

        self.Accesses = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type'] == 'Access']
        self.Knowledges = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type'] == 'Knowledge']
        self.Attack_steps = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type'] == 'Attack step']
        self.Goals = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type']== 'Goal']
        self.Skills = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type']== 'Skill']

        self.max_steps = 10


    def reset(self, g_state = False, rd_seed = 0):
        random(rd_seed)
        self.t = 0

        # Reseting the environment to the network configuration
        self.acc = {node:0 if self.nw.nodes()[node]['D'] == False else 1 for node in self.Accesses}
        self.kno = {node:0 if self.nw.nodes()[node]['D'] == False else 1 for node in self.Knowledges}
        self.goa = {node:0 if self.nw.nodes()[node]['D'] == False else 1 for node in self.Goals}
        self.ski = {node:0 if self.nw.nodes()[node]['D'] == False else 1 for node in self.Skills}
        self.act = {node:0 for node in self.Attack_steps}

        self.collection = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type'] in ['Access', 'Knowledge','Skill' , 'Goal'] and self.nw.nodes()[node]['D'] == True]

        # Reseting the environment to a given state
        if g_state:
            self.collection = []
            for node in self.Accesses:
                self.acc[node] = g_state[0][node] 
                if g_state[0][node] == 1:
                    self.collection.append(node) 
                        
            for node in self.Knowledges:
                self.kno[node] = g_state[1][node]
                if g_state[1][node] == 1:
                    self.collection.append(node)
                   
            for node in self.Skills:
                self.ski[node] = g_state[2][node]
                if g_state[2][node] == 1:
                    self.collection.append(node)  
            
            for node in self.Goals:
                self.goa[node] = g_state[3][node]
                if g_state[3][node] == 1:
                    self.collection.append(node)

        self.available_actions = self.get_available_actions()
        self.state = self.assemble_state()
        
        return self.state, self.available_actions

    
    def step(self, action, stochastic = True):
        if stochastic:
            W = self.gen_W(action)
        else:
            W = True

        cost = self.nw.nodes()[action]['C']
        if W:
            _ = 'Success'
        else:
            _ = 'Fail'

        payoff, done = self.update_state(action, W)
        
        reward = payoff - cost
        self.state = self.assemble_state()

        done = self.check_termination(done)

        self.available_actions = self.get_available_actions()
        self.t += 1
        
        return self.state, self.available_actions, reward, done, _
    
    
    def assemble_state(self):
        state = [self.acc, self.kno, self.ski, self.goa]
        return state


    def gen_W(self, action):
        W = False
        if self.nw.nodes()[action]['p'] > random():
            W = True
        return W
    
    
    def update_state(self, action, W):
        payoff = 0
        done = False
        if W:
            for edge in self.nw.edges(action):
                node = edge[1]
                if node not in self.collection:
                    node_type = self.nw.nodes()[node]['type']
                    if node_type == 'Access': 
                        self.acc[node] = 1
                    elif node_type == 'Knowledge':
                        self.kno[node] = 1
                    elif node_type == 'Goal' and self.goa[node] == 0:
                        payoff += 1
                        self.goa[node] = 1
                        done = True

                    self.collection.append(node)

            self.act[action] = 1       
        
        return payoff, done
    
    def check_termination(self, done):    
        # Number of time-steps
        if self.t > self.max_steps:
            done = True
        
        return done

    def get_available_actions(self):
        available_actions = [x for x in self.Attack_steps if any(node in self.collection for node in self.nw.predecessors(x)) and self.act[x] == 0]
        return available_actions