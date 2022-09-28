'''
Authors: 
- Juan Betancourt
- Germán Pardo
'''

import networkx as nx
from numpy.random import random, seed

class CPSsenv():

    def __init__(self, network, T_max, termination = 'One Goal', nw_params = None) -> None:
        self.termination = termination

        if network == 'SCADA':
            self.gen_SCADA_nw()
            self.set_params(nw_params)
        else:
            self.nw = network

        self.Accesses = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type'] == 'Access']
        self.Knowledges = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type'] == 'Knowledge']
        self.Attack_steps = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type'] == 'Attack step']
        self.Goals = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type']== 'Goal']
        self.Skills = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type']== 'Skill']

        self.max_steps = T_max


    def reset(self, init_type = None, init_params = None, rd_seed = 0):
        random(rd_seed)
        self.t = 0

        if init_params != None:
            self.init_network(init_type, init_params)

        # Reseting the environment to the network configuration
        self.acc = {node:0 if self.nw.nodes()[node]['D'] == False else 1 for node in self.Accesses}
        self.kno = {node:0 if self.nw.nodes()[node]['D'] == False else 1 for node in self.Knowledges}
        self.goa = {node:0 if self.nw.nodes()[node]['D'] == False else 1 for node in self.Goals}
        self.ski = {node:0 if self.nw.nodes()[node]['D'] == False else 1 for node in self.Skills}
        self.act = {node:0 for node in self.Attack_steps}

        self.collection = [node for node in self.nw.nodes() if self.nw.nodes()[node]['type'] in ['Access', 'Knowledge','Skill' , 'Goal'] and self.nw.nodes()[node]['D'] == True]

        self.available_actions = self.get_available_actions()
        self.state = self.assemble_state()
        
        return self.state, self.available_actions

   
    def step(self, action, stochastic = True):
        if stochastic:
            W = self.gen_W(action)
        else:
            W = True

        cost = self.nw.nodes()[action]['C']

        payoff = self.update_state(action, W)
        
        reward = payoff - cost
        self.state = self.assemble_state()

        done, _ = self.check_termination()

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
                        payoff += self.nw.nodes()[node]['R']
                        self.goa[node] = 1

                    self.collection.append(node)

            self.act[action] = 1       
        
        return payoff
    

    def check_termination(self): 
        _ = {'Success': False}
        done = False

        if self.termination == 'one goal':
            if self.goa[self.GOAL] == 1:
                done = True;    _['Success'] = True
        elif self.termination == 'all goals':
            any_goal = 0 in list(self.goa.values())
            if not any_goal:    
                done = True;   _['Success'] = True

        # Number of time-steps
        if self.t > self.max_steps:
            done = True

        return done, _


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
        elif action == 'a6'  and ('r2' in self.collection):     flag = True
        elif action == 'a7'  and ('s3' in self.collection and 'r5' in self.collection):     flag = True
        elif action == 'a8'  and ('r3' in self.collection and 'r5' in self.collection):     flag = True
        elif action == 'a9'  and (any(node in self.collection for node in ['r2','r3','r4','r5'])):           flag = True
        elif action == 'a10' and ('g1' in self.collection):                                 flag = True
        elif action == 'a11' and ('g1' in self.collection):                                 flag = True
        elif action == 'a12' and ('k2' in self.collection and 'r6' in self.collection):     flag = True
        elif action == 'a13' and ('r6' in self.collection):                                 flag = True
        elif action == 'a14' and ('k3' in self.collection and 'r6' in self.collection):     flag = True
        elif action == 'a15' and ('k4' in self.collection and 'r7' in self.collection):     flag = True
        elif action == 'a16' and ('s3' in self.collection and 'r8' in self.collection):     flag = True
        elif action == 'a17' and ('r8' in self.collection):                                 flag = True 
        elif action == 'a18' and ('g1' in self.collection):                                 flag = True

        return flag


        # if action ==   'a1'  and ('r1' in self.collection and 'k1' in self.collection):      flag = True
        # elif action == 'a2'  and ('r1' in self.collection and 's1' in self.collection):      flag = True
        # elif action == 'a3'  and ('s2' in self.collection and any(node in self.collection for node in ['r1', 'r3', 'r4'])): flag = True
        # elif action == 'a4'  and ('s2' in self.collection and any(node in self.collection for node in ['r1', 'r2', 'r4'])): flag = True
        # elif action == 'a5'  and ('s2' in self.collection and any(node in self.collection for node in ['r1', 'r2', 'r3'])): flag = True
        # elif action == 'a6'  and ('s2' in self.collection and 'r2' in self.collection):     flag = True
        # elif action == 'a7'  and ('s3' in self.collection and 'r5' in self.collection):     flag = True
        # elif action == 'a8'  and ('s3' in self.collection and 'r5' in self.collection):     flag = True
        # elif action == 'a9'  and (any(node in self.collection for node in ['r2','r3','r4','r5'])):           flag = True
        # elif action == 'a10' and ('g1' in self.collection):                                 flag = True
        # elif action == 'a11' and ('g1' in self.collection):                                 flag = True
        # elif action == 'a12' and ('k2' in self.collection and 'r6' in self.collection):     flag = True
        # elif action == 'a13' and ('r6' in self.collection):                                 flag = True
        # elif action == 'a14' and ('k3' in self.collection and 'r6' in self.collection):     flag = True
        # elif action == 'a15' and ('k4' in self.collection and 'r7' in self.collection):     flag = True
        # elif action == 'a16' and ('s3' in self.collection and 'r8' in self.collection):     flag = True
        # elif action == 'a17' and ('r8' in self.collection):                                 flag = True 
        # elif action == 'a18' and ('g1' in self.collection):                                 flag = True

        # return flag


    def gen_SCADA_nw(self):
        '''
        SCADA System example
        '''
        network = nx.DiGraph()

        nodes = [
            ('r1', {'type': 'Access', 'D': False}),  ('r2', {'type': 'Access', 'D': False}), ('r3', {'type': 'Access', 'D': False}), 
            ('r4', {'type': 'Access', 'D': False}), ('r5', {'type': 'Access', 'D': False}), ('r6', {'type': 'Access', 'D': False}),
            ('r7', {'type': 'Access', 'D': False}), ('r8', {'type': 'Access', 'D': False}),
            
            ('k1', {'type': 'Knowledge', 'D': False}), ('k2', {'type': 'Knowledge', 'D': False}),
            ('k3', {'type': 'Knowledge', 'D': False}), ('k4', {'type': 'Knowledge', 'D': False}),
            
            ('s1', {'type': 'Skill', 'D': False}), ('s2', {'type': 'Skill', 'D': False}), ('s3', {'type': 'Skill', 'D': False}),
            
            ('g1', {'type': 'Goal', 'D': False, 'R': 0}), ('g2', {'type': 'Goal', 'D': False, 'R': 0}), ('g3', {'type': 'Goal', 'D': False, 'R': 0}), 
            ('g4', {'type': 'Goal', 'D': False, 'R': 0}), ('g5', {'type': 'Goal', 'D': False, 'R': 0}),
            
            ('a1',  {'type': 'Attack step', 'D': False, 'p': 0.7, 'C': 50}), ('a2',  {'type': 'Attack step', 'D': False, 'p': 0.3, 'C': 60}),
            ('a3',  {'type': 'Attack step', 'D': False, 'p': 0.5, 'C': 60}), ('a4',  {'type': 'Attack step', 'D': False, 'p': 0.1, 'C': 120}),
            ('a5',  {'type': 'Attack step', 'D': False, 'p': 0.3, 'C': 90}), ('a6',  {'type': 'Attack step', 'D': False, 'p': 0.6, 'C': 30}),
            ('a7',  {'type': 'Attack step', 'D': False, 'p': 0.75,'C': 70}), ('a8',  {'type': 'Attack step', 'D': False, 'p': 0.8, 'C': 20}),
            ('a9',  {'type': 'Attack step', 'D': False, 'p': 0.95,'C': 10}), ('a10', {'type': 'Attack step', 'D': False, 'p': 0.5, 'C': 90}),
            ('a11', {'type': 'Attack step', 'D': False, 'p': 0.85,'C': 20}), ('a12', {'type': 'Attack step', 'D': False, 'p': 1,   'C': 1}),
            ('a13', {'type': 'Attack step', 'D': False, 'p': 0.7, 'C': 10}), ('a14', {'type': 'Attack step', 'D': False, 'p': 1,   'C': 1}),    
            ('a15', {'type': 'Attack step', 'D': False, 'p': 1,   'C': 1}),  ('a16', {'type': 'Attack step', 'D': False, 'p': 0.65,'C': 90}),
            ('a17', {'type': 'Attack step', 'D': False, 'p': 0.75,'C': 150}),('a18', {'type': 'Attack step', 'D': False, 'p': 0.5, 'C': 20})
                ]

        for node in nodes:
            if node[1]['type'] == 'Attack step':
                node[1]['C'] /= 200
        

        network.add_nodes_from(nodes)

        edges = [('r1','a1'), ('r1','a2'), ('r1','a3'), ('r1','a4'), ('r1','a5'), ('k1','a1'), ('k2','a12'),('k3','a14'), ('k4','a15'), ('s1','a2'),
                ('s2','a3'), ('s2','a4'), ('s2','a5'), ('s3','a16'),('s3','a7'), ('a1','r2'), ('a2','r2'), ('a3','r2'),  ('a4','r3'),  ('a5','r4'),
                ('r2','a4'), ('r2','a5'), ('r2','a6'), ('r2','a9'), ('r3','a3'), ('r3','a5'), ('r3','a8'), ('r3','a9'),
                
                ('r6','a12'),('r6','a14'),('r3','a13'),('r7','a15'),('r8','a16'), ('r8','a17'),('r4','a3'), ('r4','a4'), ('r4','a9'), ('a6','r5'),
                ('r5','a7'), ('r5','a8'), ('r5','a9'), ('a7','g1'), ('a8','g2'),  ('a9','g3'), ('g1','a10'),('g1','a11'),('g1','a18'),
                ('a10','g4'),('a11','g2'),('a12','r2'),('a13','r7'),('a14','r5'), ('a15','r8'),('a16','g1'),('a17','g5'),('a18','g5'),]

        network.add_edges_from(edges)

        self.nw = network


    def set_params(self, params):
        if 'Probs' in params.keys():
            for att,prob in params['Probs'].items():
                self.nw.nodes()[att]['p'] = prob

        if 'Rewards' in params.keys():
            for goal in params['Rewards']:
                self.nw.nodes()[goal]['R'] = 2
                if len(params['Rewards']) == 1:
                    self.GOAL = goal
                # TODO IMPLEMENT LIST OF DESIRED GOALS WHEN 
        # TODO IMPLEMENT ELSE: AT LEAST ONE GOAL HAS REWARD
        
        

    def init_network(self, init_type, *args):

        # Environment's state format
        if init_type == 'env state':
            init_state = args[0]
            for access in init_state[0].keys():
                self.nw.nodes()[access]['D'] = bool(init_state[0][access])
            for know in init_state[1].keys():
                self.nw.nodes()[know]['D'] = bool(init_state[1][know])
            for skill in init_state[2].keys():
                self.nw.nodes()[skill]['D'] = bool(init_state[2][skill])
            for goal in init_state[3].keys():
                self.nw.nodes()[goal]['D'] = bool(init_state[3][goal])

        elif init_type == 'active list':
            active = args[0]
            for node in active:
                self.nw.nodes()[node]['D'] = True
            

        # else:
        #     if 'Access' in init.keys():
        #         for access in init['Access']:
        #             self.nw.nodes()[access]['D'] = True
        #     if 'Skills' in init.keys():
        #         for skill in init['Skills']:
        #             self.nw.nodes()[skill]['D'] = 1
        #     if 'Knowledges' in init.keys():
        #         for know in init['Knowledges']:
        #             self.nw.nodes()[know]['D'] = 1
        #     if 'Goals' in init.keys():
        #         for goal in init['Goals']:
        #             self.nw.nodes()[goal]['D'] = 1

            # # Reseting the environment to a given state
            # if g_state:
            # self.collection = []
            # for node in self.Accesses:
            #     self.acc[node] = g_state[0][node] 
            #     if g_state[0][node] == 1:
            #         self.collection.append(node) 
                        
            # for node in self.Knowledges:
            #     self.kno[node] = g_state[1][node]
            #     if g_state[1][node] == 1:
            #         self.collection.append(node)
                   
            # for node in self.Skills:
            #     self.ski[node] = g_state[2][node]
            #     if g_state[2][node] == 1:
            #         self.collection.append(node)  
            
            # for node in self.Goals:
            #     self.goa[node] = g_state[3][node]
            #     if g_state[3][node] == 1:
            #         self.collection.append(node)
        