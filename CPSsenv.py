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

        self.max_steps = 7


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

        payoff, done, _ = self.update_state(action, W)
        
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
        _ = 0
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
                        _ = True

                    self.collection.append(node)

            self.act[action] = 1       
        
        return payoff, done, _
    
    
    def check_termination(self, done):    
        # Number of time-steps
        if self.t > self.max_steps:
            done = True
        
        return done


    def get_available_actions(self):
        available_actions = [x for x in self.Attack_steps if self.eval_predecessors(x) and self.act[x] == 0]
        return available_actions
    
    # TODO Generalization
    def eval_predecessors(self, action):
        flag = False
        if action ==   'a1'  and ('r1' in self.collection and 'k1' in self.collection):      flag = True
        elif action == 'a2'  and ('r1' in self.collection and 's1' in self.collection):      flag = True
        elif action == 'a3'  and ('s2' in self.collection and any(node in self.collection for node in ['r1', 'r3', 'r4'])): flag = True
        elif action == 'a4'  and ('s2' in self.collection and any(node in self.collection for node in ['r1', 'r2', 'r4'])): flag = True
        elif action == 'a5'  and ('s2' in self.collection and any(node in self.collection for node in ['r1', 'r2', 'r3'])): flag = True
        elif action == 'a6'  and ('s2' in self.collection and 'r2' in self.collection):     flag = True
        elif action == 'a7'  and ('s3' in self.collection and 'r5' in self.collection):     flag = True
        elif action == 'a8'  and ('s3' in self.collection and 'r5' in self.collection):     flag = True
        elif action == 'a9'  and (any(node in self.collection for node in ['r2','r3','r4','r5'])):           flag = True
        elif action == 'a10' and ('g1' in self.collection):                                 flag = True
        elif action == 'a11' and ('g1' in self.collection):                                 flag = True
        elif action == 'a12' and ('k2' in self.collection and 'r6' in self.collection):     flag = True
        elif action == 'a13' and ('r6' in self.collection):                                 flag = True
        elif action == 'a14' and ('k3' in self.collection and 'r6' in self.collection):     flag = True
        elif action == 'a15' and ('k4' in self.collection and 'r7' in self.collection):     flag = True
        elif action == 'a16' and ('s3' in self.collection and 'r8' in self.collection):     flag = True
        elif action == 'a17' and ('r8' in self.collection):                                 flag = True 
        elif action == 'a18' and ('g1' in self.colleciton):                                 flag = True

        return flag