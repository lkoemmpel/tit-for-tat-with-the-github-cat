'''-------------------
      PACKAGES
-------------------'''

import networkx as nx
import random
import matplotlib.pyplot as plt
import math
#import numpy as np

'''-------------------
      IMPORTS
-------------------'''
import graph_initialization as init

'''-------------------
IMPORTANT VARIABLES
-------------------'''
B=2
C=1
this_lambda=0.5
kappa=0.5
val0=0.5
matrix=[ [(B-C, B-C), (-C, B)] , [(B, -C), (0,0)] ]
coop_index={'Cooperate':0, 'Defect':1}

'''-------------------
HELPER FUNCTIONS
-------------------'''

def F(val):
     #sigmoid logictic function used to determine 
     #next-round probabilities
     return (1+math.exp(-kappa*(val-val0)))**(-1)

def bernoulli(p):
     #random variable with parameter p
     if random.random()<=p:
          return 1
     return 0

def action(G, v, noise):
  #noise=Pr[node does not do intended action]
  #changes strategy with Pr=noise
  strat=G.node[v]['strategy']
  if random.random()<=noise:
    if strat=='Cooperate':
      strat='Defect'
    elif strat=='Defect':
      strat='Cooperate'
  return coop_index[strat]

'''-------------------
KINDS OF INTERACTION PROCESSES
-------------------'''

def general_reciprocity_simple(G, set_nodes, b, c, f, asynchronous=False): 
  #realize all of the interactions between pairs of players belonging to the set
  for v in set_nodes:
    requested=random.choice(list(G.adj[v].keys()))

    #interact1(v,w): If the node w decides to help (with probability p_w(t)
    if random.random()<=G.node[requested]['coop-state']:
      #update payoffs, fitness, probabilities, etc for next timestep
      G.node[requested]['turn-payoff']-=c
      G.node[v]['turn-payoff']+=b
      G.node[v]['total-payoff']+=G.node[v]['turn-payoff']
      if asynchronous:
        A=(G.node[v]['coop-state'])**(1-this_lambda)
        B=(f(G.node[v]['total-payoff']))**(this_lambda)
        G.node[v]['coop-state']=A*B
      else:
        G.node[v]['coop-state']=f(G.node[v]['total-payoff'])
        G.node[v]['turn-payoff']=0
  #return modified graph 
  return G

def general_reciprocity_bernoulli(self, set_nodes, b, c, f, asynchronous=False):    G=self.graph
  #realize all of the interactions between pairs of players belonging to the set
  for v in set_nodes:
    requested=random.choice(G.neighbors(v))
    #set x for node v
    G.node[v]['x']=bernoulli(G.node[v]['coop-state'])
    x_v=G.node[v]['x']
    #set x for requested node
    G.node[requested]['x']=bernoulli(G.node[requested]['coop-state'])
    x_req=G.node[requested]['x']
    #sum bernoullis for rho values
    sum_rhos=0
    for k in G.adj[v].keys():
      deg_k=len(G.adj[k])
      sum_rhos+=bernoulli(deg_k**(-1))
      #now, set the turn payoff value that was determined stochastically
      G.node[v]['turn-payoff']=b*x_v-c*x_req*sum_rhos
      #b*x_requested-c*x_v*sum_(nbhd of i)[rho_k at t]
      if asynchronous:
        A=(G.node[v]['coop-state'])**(1-this_lambda)
        B=(f(G.node[v]['total-payoff']))**(this_lambda)
        G.node[v]['coop-state']=A*B
      else:
        G.node[v]['coop-state']=f(G.node[v]['total-payoff'])
        G.node[v]['turn-payoff']=0
  #return modified graph 
  return G

def interaction_BD(G, payoff_mtx, delta=0, noise=0):
  #Simulation of the process where every edge interacts with
  #some randomly chosen set of its neighbors
  record=set()
  for v in nx.nodes(G):
    for w in G.neighbors(v):
      occurs=random.choice([True,False])
      #does interaction occur?
      if occurs:
        #they haven't interacted yet
        if (w,v) not in record:
          action_v=action(G, v, noise)
          action_w=action(G, w,noise)
          record.add((v,w))
          G.node[v]['payoffs'].append(payoff_mtx[action_v][action_w][0])
          G.node[w]['payoffs'].append(payoff_mtx[action_v][action_w][1])
      if len(G.node[v]['payoffs'])==0:
        G.node[v]['fitness']=1
      else:
        G.node[v]['fitness']=1+delta*sum(G.node[v]['payoffs'])/len(G.node[v]['payoffs'])
        G.node[v]['payoffs']=[]
  return G
     
'''-------------------
GRAPH FOR TESTING
-------------------'''

strat_list=['Cooperate', 'Defect']
G=init.generate_lattice(3,4)
init.label_birth_death(G)
init.label_utkovski(G)


'''-------------------
TESTS
-------------------'''

# T=4
# for t in range(T):

#      g=interaction(G)
#      set_nodes=G.adj.keys()
#      new_graph = g.general_reciprocity_simple(set_nodes, B, C, F, False)
#      # Creates picture of graph 
#      nx.draw(G,with_labels=True)
#      plt.show()

#stuff for testing
'''
T=4
b = 2
c = 1
matrix = [[b-c, -c], [b, 0]]

for t in range(T):
  g=interaction_process(G, b, c)
  set_nodes=G.adj.keys()
  new_graph = g.interaction_BD(matrix, 0, 0.2)
  # Creates picture of graph 
  D1=nx.get_node_attributes(G, 'strategy')
  D2=nx.get_node_attributes(G, 'fitness')
  print('--------')
  print(D1)
  print('***')
  print(D2)
  print('--------')
  nx.draw(G,with_labels=True)
  plt.show()
'''





          


