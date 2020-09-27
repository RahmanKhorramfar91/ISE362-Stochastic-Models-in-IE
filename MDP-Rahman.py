# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:52:08 2020

@author: Rahman Khorramfar
"""

import numpy as np;import sys
import pandas as pd; import csv;
#%% Problm Parameters

if len(sys.argv)>=3:   
    gamma = float(sys.argv[1]);# the discount factor
    epsilon = float(sys.argv[2]); # error tolerance
elif len(sys.argv)>=2:
    gamma = float(sys.argv[1]);
else:
    gamma = 0.8;
    epsilon = 0.001;


    
#%% Read the probability transition data (S*S*A), and rewad data (S*A)
Transitions = {}; # dictionary
Reward = {}; # dictionary

with open('transitions.csv', 'r') as csvfile:
    #reader = pd.read_csv(csvfile,delimiter=',');
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[0] in Transitions:
            if row[1] in Transitions[row[0]]:
                Transitions[row[0]][row[1]].append((float(row[3]), row[2]))
            else:
                Transitions[row[0]][row[1]] = [(float(row[3]), row[2])]
        else:
            Transitions[row[0]] = {row[1]:[(float(row[3]),row[2])]}

    #read rewards file and save it to a variable
with open('rewards.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        Reward[row[0]] = float(row[1]) if row[1] != 'None' else None  
#%% Value iteration: Main loop
"""
Solving the MDP by value iteration.
returns utility values for states after convergence
"""
states =  Transitions.keys();
#actions = mdp.actions
#print(states); print(actions);
#initialize value of all the states to 0 (this is k=0 case)
V1 = {s: 0 for s in states}
while True:
    V = V1.copy()
    delta = 0
    for s in states:
        #Bellman update, update the utility values
        V1[s] = Reward[s] + gamma * max([ sum([p * V[s1] for (p, s1)
        in Transitions[s][a]]) for a in Transitions[s].keys()]);
        #calculate maximum difference in value
        delta = max(delta, abs(V1[s] - V[s]))

    #check for convergence, if values converged then return V
    if delta < epsilon * (1 - gamma) / gamma:
        break;
        
        
#%% Post solution analysis   
pi = {}
for s in states:
    pi[s] = max(Transitions[s], key=lambda a: sum([p * V[s1] for (p, s1) in Transitions[s][a]])); 
       
print( 'State - Value')
for s in V:
    print( s, ' - ' , V[s]);
print ('\nOptimal policy is \nState - Action')
for s in pi:
    print( s, ' - ' , pi[s]);



