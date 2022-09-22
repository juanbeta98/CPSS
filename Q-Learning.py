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

rd_seed = 0
seed(rd_seed)

'''
SCADA System example
'''
network = nx.DiGraph()

nodes = [('r1', {'type': 'Access', 'D': True}),  ('r2', {'type': 'Access', 'D': False}), ('r3', {'type': 'Access', 'D': False}), 
         ('r4', {'type': 'Access', 'D': False}), ('r5', {'type': 'Access', 'D': False}), ('r6', {'type': 'Access', 'D': False}),
         ('r7', {'type': 'Access', 'D': False}), ('r8', {'type': 'Access', 'D': False}),
         
         ('k1', {'type': 'Knowledge', 'D': False}), ('k2', {'type': 'Knowledge', 'D': False}),
         ('k3', {'type': 'Knowledge', 'D': False}), ('k4', {'type': 'Knowledge', 'D': False}),
         
         ('s1', {'type': 'Skill', 'D': False}), ('s2', {'type': 'Skill', 'D': False}), ('s3', {'type': 'Skill', 'D': False}),
         
         ('g1', {'type': 'Goal', 'D': False, 'R': 1}), ('g2', {'type': 'Goal', 'D': False, 'R': 1}), ('g3', {'type': 'Goal', 'D': False, 'R': 1}), 
         ('g4', {'type': 'Goal', 'D': False, 'R': 1}), ('g5', {'type': 'Goal', 'D': False, 'R': 1}),
         
         ('a1',  {'type': 'Attack step', 'D': False, 'p': 1,   'C': 50}), ('a2',  {'type': 'Attack step', 'D': False, 'p': 0.3, 'C': 60}),
         ('a3',  {'type': 'Attack step', 'D': False, 'p': 0.5, 'C': 60}), ('a4',  {'type': 'Attack step', 'D': False, 'p': 0.1, 'C': 120}),
         ('a5',  {'type': 'Attack step', 'D': False, 'p': 0.3, 'C': 90}), ('a6',  {'type': 'Attack step', 'D': False, 'p': 0.6, 'C': 30}),
         ('a7',  {'type': 'Attack step', 'D': False, 'p': 0.75,'C': 70}), ('a8',  {'type': 'Attack step', 'D': False, 'p': 0.8, 'C': 20}),
         ('a9',  {'type': 'Attack step', 'D': False, 'p': 0.95,'C': 10}), ('a10', {'type': 'Attack step', 'D': False, 'p': 0.5, 'C': 90}),
         ('a11', {'type': 'Attack step', 'D': False, 'p': 0.85,'C': 20}), ('a12', {'type': 'Attack step', 'D': False, 'p': 1,   'C': 1}),
         ('a13', {'type': 'Attack step', 'D': False, 'p': 0.7, 'C': 10}), ('a14', {'type': 'Attack step', 'D': False, 'p': 1,   'C': 1}),    
         ('a15', {'type': 'Attack step', 'D': False, 'p': 1,   'C': 1}),  ('a16', {'type': 'Attack step', 'D': False, 'p': 0.65,'C': 90}),
         ('a17', {'type': 'Attack step', 'D': False, 'p': 0.75,'C': 150}),('a18', {'type': 'Attack step', 'D': False, 'p': 0.5, 'C': 20})]

for node in nodes:
    if node[1]['type'] == 'Attack step':
        node[1]['C'] /= 300

network.add_nodes_from(nodes)

edges = [('r1','a1'), ('r1','a2'), ('r1','a3'), ('r1','a4'), ('r1','a5'), ('k1','a1'), ('k2','a12'),('k3','a14'), ('k4','a15'), ('s1','a2'),
         ('s2','a3'), ('s2','a4'), ('s2','a5'), ('s3','a16'),('s3','a7'), ('a1','r2'), ('a2','r2'), ('a3','r2'),  ('a4','r3'),  ('a5','r4'),
         ('r2','a4'), ('r2','a5'), ('r2','a6'), ('r2','a9'), ('r3','a3'), ('r3','a5'), ('r3','a8'), ('r3','a9'),
         
         ('r6','a12'),('r6','a14'),('r3','a13'),('r7','a15'),('r8','a16'), ('r8','a17'),('r4','a3'), ('r4','a4'), ('r4','a9'), ('a6','r5'),
         ('r5','a7'), ('r5','a8'), ('r5','a9'), ('a7','g1'), ('a8','g2'),  ('a9','g3'), ('g1','a10'),('g1','a11'),('g1','a18'),
         ('a10','g4'),('a11','g2'),('a12','r2'),('a13','r7'),('a14','r5'), ('a15','r8'),('a16','g1'),('a17','g5'),('a18','g5'),]

network.add_edges_from(edges)

def translate_state(state):
    trans_state = []
    for j in range(3):
        for i in state[j].values():
            trans_state.append(i)
    return trans_state


'''
Q-Learning tranining
'''
num_replications = 20
Replications = range(num_replications)

### Q-Learning
alpha = 0.1         # How fast does the agent learn
gamma = 0.95             # How important are future actions

Episodes = 30000            # Number of episodes

epsilon = 0.7                    # Rate at which random actions will be 
start_e_decaying = 1             # First episode at which decay epsilon
end_e_decaying = Episodes // 2   # Last episode at which decay epsilon
epsilon_decay_value = epsilon / (end_e_decaying - start_e_decaying)     # Amount of decayment of epsilon   

episodes_rewards = {}
succeses = {}

env = CPSsenv(network)

for replica in Replications:

    print('Repcation n:',replica)
    q_table = {}
    episodes_rewards[replica] = []
    succeses[replica] = []
    for episode in range(Episodes):

        episode_reward = 0
        state, available_actions = env.reset(rd_seed = rd_seed + episode * (1 + replica))
        sttate = translate_state(state)
        done = False

        while not done:
            if random() < epsilon or tuple(sttate) not in list(q_table.keys()):
                action = choice(available_actions)
            elif {i:j for i,j in q_table[tuple(sttate)].items() if j in available_actions} == {}:
                action = choice(available_actions)
            else:
                real_dict = {i:j for i,j in q_table[tuple(sttate)].items() if j in available_actions}
                action = max(real_dict, key = real_dict.get)

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
            
        episodes_rewards[replica].append(episode_reward)
        succeses[replica].append(_)

'''
Mobiled-averaged rewards for plotting 
'''
average_episodes = 500
averaged_rewards = []
for episode in range(Episodes):
    if episode <= average_episodes/2:
        av_rw = 0
        averaged_rewards.append(sum(episodes_rewards[replica][:average_episodes] for replica in Replications)/average_episodes)
    elif episode <= Episodes - average_episodes/2:
        lower = int(episode-average_episodes/2)
        upper = int(episode+average_episodes/2)
        averaged_rewards.append(sum(episodes_rewards[replica][lower:upper] for replica in Replications)/average_episodes)
    else:
        averaged_rewards.append(sum(episodes_rewards[replica][Episodes - average_episodes:] for replica in Replications)/average_episodes)

print(sum(succeses))
plt.plot(averaged_rewards, color = 'purple')
plt.title('Average reward through the episodes')
plt.xlabel('Episodes')
plt.ylabel('Average reward')
plt.show()

'''
Ploting succesfull episodes
'''
probs = [sum(succeses[episode])/num_replications for episode in Episodes]
plt.plot(probs, color = 'blue')
plt.title('Succesfull attacks through the episodes')
plt.xlabel('Episodes')
plt.ylabel('Proportion of succesfull episodes')
plt.show()


'''
Q-Learning training
'''
for episode in range(Episodes):

    episode_reward = 0
    state, available_actions = env.reset(rd_seed = rd_seed + episode)
    sttate = translate_state(state)
    done = False

    while not done:
        if random() < epsilon or tuple(sttate) not in list(q_table.keys()):
            action = choice(available_actions)
        elif {i:j for i,j in q_table[tuple(sttate)].items() if j in available_actions} == {}:
            action = choice(available_actions)
        else:
            real_dict = {i:j for i,j in q_table[tuple(sttate)].items() if j in available_actions}
            action = max(real_dict, key = real_dict.get)

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
    succeses.append(_)

'''
Q-Learining testing
'''
print('\n \n########## Q-LEARNING TESTING ##########')
### Q-Learning
env = CPSsenv(network)

episode_reward = 0
state, available_actions = env.reset(rd_seed = rd_seed + 1)
sttate = translate_state(state)
done = False

while not done:

    real_dict = {i:j for i,j in q_table[tuple(sttate)].items()}
    action = max(real_dict, key = real_dict.get)

    print(f't = {env.t} \t \t Action = {action}')

    new_state, available_actions, reward, done, _ = env.step(action)
    episode_reward += reward
    new_sttate = translate_state(new_state)
        
    state = new_state
    sttate = new_sttate

print('\n \n')

