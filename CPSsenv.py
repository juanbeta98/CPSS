'''
Author: Juan Betancourt 
'''

import networkx as nx
from numpy.random import random


class CPSsenv():

    def __init__(self, network) -> None:

        self.nw = network

        self.Accesses = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type'] == 'Access']
        self.Knowledges = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type'] == 'Knowledge']
        self.Attack_steps = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type'] == 'Attack step']
        self.Goals = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type']== 'Goal']
        self.Skills = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type']== 'Skill']

        self.max_steps = 30



    def reset(self, l, random_start = False):
        self.t = 0

        
            
        
        self.acc = {node:0 if self.nw.nodes()[node]['D'] == False else 1 for node in self.Accesses}
        self.kno = {node:0 if self.nw.nodes()[node]['D'] == False else 1 for node in self.Knowledges}
        self.goa = {node:0 if self.nw.nodes()[node]['D'] == False else 1 for node in self.Goals}
        self.ski = {node:0 if self.nw.nodes()[node]['D'] == False else 1 for node in self.Skills}
        self.act = {node:0 for node in self.Attack_steps}

        self.collection = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type'] in ['Access', 'Knowledge','Skill' , 'Goal'] and self.nw.nodes()[node]['D'] == True]

        
                
        ### l recibe un estado y lo actualiza en el ambiente
        if l:
            
            for node in self.Accesses:
                    self.acc[node] = l[0][node]
                    
                    if l[0][node] == 1:
                        self.collection.append(node)
                     #   self.nw.nodes()[node]['D'] = True
                        
                        
            for node in self.Knowledges:
                    self.kno[node] = l[1][node]
                    
                    if l[1][node] == 1:
                        self.collection.append(node)
                    #    self.nw.nodes()[node]['D'] = True
                    
                    
                   
            for node in self.Skills:
                    self.ski[node] = l[2][node]
                    
                    if l[2][node] == 1:
                        self.collection.append(node)
                    #    self.nw.nodes()[node]['D'] = True
                        
            
            for node in self.Goals:
                    self.goa[node] = l[3][node]
                    
                    if l[3][node] == 1:
                        self.collection.append(node)
                    #    self.nw.nodes()[node]['D'] = True
        
        
        if random_start: 
            for access in self.Accesses:
                if random() < 0.5:
                    self.acc[access] = 1
                    self.collection.append(access)
            for knowledge in self.Knowledges:
                if random() < 0.5:
                    self.kno[knowledge] = 1
                    self.collection.append(knowledge)
            for skill in self.Skills:
                if random() < 0.5:
                    self.ski[skill] = 1
                    self.collection.append(skill)

        self.available_actions = self.get_available_actions()
        self.state = self.assemble_state()
        
        return self.state, self.available_actions

    
    
    ### Función step para Q-learning
    
    def step(self, action):
        W = self.gen_W(action)
        cost = self.nw.nodes()[action]['C']
        _ = ''

        payoff = self.update_state(action, W)
        
        reward = payoff - cost
        self.state = self.assemble_state()

        done = self.check_termination()

        self.available_actions = self.get_available_actions()
        self.t += 1
        
        return self.state, self.available_actions, reward, done, _
    
    
    
    ### Función step para vaule iteration
    
    def step2(self, action):
        W = True
        cost = self.nw.nodes()[action]['C']
        _ = ''

        payoff = self.update_state(action, W)
        
        reward = payoff - cost
        self.state = self.assemble_state()

        done = self.check_termination()

        self.available_actions = self.get_available_actions()
        self.t += 1
        
        return self.state, self.available_actions, reward, done, _
    
    
    
    

    def assemble_state(self):
        state = [self.acc, self.kno, self.ski, self.goa]
        return state

    
    ### Función gen_W para Q-learning
    
    def gen_W(self, action):
        W = False
        if self.nw.nodes()[action]['p'] > random():
            W = True
        return W
    
    ### Función gen_W para value iteration
    def gen_W2(self, action):
        W = False
        if self.nw.nodes()[action]['p'] > 0: 
            W = True
        return W
    
    def update_state(self, action, W):
        payoff = 0
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
                        
                    self.collection.append(node)
            self.act[action] = 1       
        
        return payoff
    
    def check_termination(self):
        done = 0 not in self.goa.values()
    
        # Number of time-steps
        if self.t > self.max_steps:
            done = True
        
        return done

    def get_available_actions(self):
        available_actions = [x for x in self.Attack_steps if any(node in self.collection for node in self.nw.predecessors(x)) and self.act[x] == 0]
        return available_actions