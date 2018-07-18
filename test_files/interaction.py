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
this_lambda=0.5
kappa=0.5
val0=0.5
coop_index={'Cooperate':0, 'Defect':1}
strat_index={0:'Cooperate', 1:'Defect'}

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

def action(G, intention_v, noise):
  #noise=Pr[node does not do intended action]
  #changes strategy with Pr=noise
  action=intention_v
  if random.random()<=noise:
    if action=='Cooperate':
      action='Defect'
    elif action=='Defect':
      action='Cooperate'
  return coop_index[action]

def intention(G, v):
  if G.node[v]['strategy'] == 'Cooperate':
    return 'Cooperate'
  elif G.node[v]['strategy'] == 'Defect':
    return 'Defect'

def normalization(val):
    V=1+math.exp(-val)
    #return 1/V
    return val

def f1(x):
  return x**2

def Theta_paper1(val, delta):
  return math.exp(delta*val)

'''-------------------
KINDS OF INTERACTION PROCESSES
-------------------'''

def general_reciprocity_simple(G, set_nodes, b, c, f, asynchronous=False): 
  num_help_actions=0
  helpers=set()
  #realize all of the interactions between pairs of players belonging to the set
  for v in set_nodes:
    requested_to=random.choice(list(G.adj[v].keys()))

    #interact1(v,w): If the node w decides to help (with probability p_w(t)
    if random.random()<=G.node[requested_to]['coop_state']:
      num_help_actions+=1
      helpers.add(requested_to)
      #update payoffs, fitness, probabilities, etc for next timestep
      G.node[requested_to]['turn_payoff']-=c
      G.node[v]['turn_payoff']+=b
  for v in set_nodes:
    G.node[v]['total_payoff']+=G.node[v]['turn_payoff']
  if asynchronous:
    A=(G.node[v]['coop_state'])**(1-this_lambda)
    B=(F(G.node[v]['total_payoff']))**(this_lambda)
    G.node[v]['coop_state']=A*B
  else:
    G.node[v]['coop_state']=F(G.node[v]['total_payoff'])
  G.node[v]['turn_payoff']=0
  #return modified graph 
  return G, helpers, num_help_actions

def general_reciprocity_bernoulli(self, set_nodes, b, c, asynchronous=False):    
  #realize all of the interactions between pairs of players belonging to the set
  for v in set_nodes:
    requested=random.choice(G.neighbors(v))
    #set x for node v
    G.node[v]['x']=bernoulli(G.node[v]['coop_state'])
    x_v=G.node[v]['x']
    #set x for requested node
    G.node[requested]['x']=bernoulli(G.node[requested]['coop_state'])
    x_req=G.node[requested]['x']
    #sum bernoullis for rho values
    sum_rhos=0
    for k in G.adj[v].keys():
      deg_k=len(G.adj[k])
      sum_rhos+=bernoulli(deg_k**(-1))
      #now, set the turn payoff value that was determined stochastically
      G.node[v]['turn_payoff']=b*x_v-c*x_req*sum_rhos
      #b*x_requested-c*x_v*sum_(nbhd of i)[rho_k at t]
      if asynchronous:
        A=(G.node[v]['coop_state'])**(1-this_lambda)
        B=(f(G.node[v]['total_payoff']))**(this_lambda)
        G.node[v]['coop_state']=A*B
      else:
        G.node[v]['coop_state']=f(G.node[v]['total_payoff'])
        G.node[v]['turn_payoff']=0
  #return modified graph 
  return G

def interaction_BD(G, payoff_mtx, delta=0, noise=0):
  #Simulation of the process where every edge interacts with
  #some randomly chosen set of its neighbors
  print('INTERACTION')
  print('---------')
  record=set()
  for v in nx.nodes(G):
    for w in G.neighbors(v):
      print("Examining node ", v, " and it's neighbor ", w)
      #occurs=random.choice([True,False])
      occurs=random.choice([True])
      #does interaction occur?
      if occurs:
        print(v, " and ", w, " were chosen to interact!")
        #they haven't interacted yet
        if (w,v) not in record:
          print("They have not interacted before")
          action_v=action(G, intention(G,v), noise)
          action_w=action(G, intention(G,w), noise)
          print('action of v is: ' + strat_index[action_v])
          print('action of w is: ' + strat_index[action_w])
          record.add((v,w))
          G.node[v]['payoffs'].append(payoff_mtx[action_v][action_w][0])
          G.node[w]['payoffs'].append(payoff_mtx[action_v][action_w][1])
    if len(G.node[v]['payoffs']) != 0:
      old=G.node[v]['fitness']
      avg_payoff=sum(G.node[v]['payoffs'])/len(G.node[v]['payoffs'])
      #max_deg=max([G.degree[n] for n in G.nodes()])
      #G.node[v]['fitness']=normalization2(sum(G.node[v]['payoffs']), delta, max_p, min_p, max_deg)
      new=normalization(Theta_paper1(avg_payoff, delta))
      G.node[v]['fitness']=(old+new)/2
      print('Fitness changed from ' +str(old)+ ' to '+ str(G.node[v]['fitness']))
      #restart payoff list for next round
      G.node[v]['payoffs']=[]
  return G

def interaction_depends_fitness(G, payoff_mtx, delta=0, noise=0, function=f1):
  D={}
  #D[(A,B)]=whether A cooperates with B or not
  for a in nx.nodes(G):
    for b in G.neighbors(a):
      if random.random()<function(G.node[b]['fitness']):
        D[(a,b)]='Defect'
      else:
        D[(a,b)]='Cooperate'
  for a in nx.nodes(G):
    for b in G.neighbors(a):
      action_a=action(G, D[(a,b)], noise)
      action_b=action(G, D[(b,a)], noise)
      G.node[a]['payoffs'].append(payoff_mtx[action_a][action_b][0])
      G.node[b]['payoffs'].append(payoff_mtx[action_a][action_b][1])
  for v in nx.nodes(G):
    if len(G.node[v]['payoffs']) != 0:
      G.node[v]['fitness']=normalization(Theta_paper1(sum(G.node[v]['payoffs'])/len(G.node[v]['payoffs']), delta))
      G.node[v]['payoffs']=[]
  return G

def interaction_some_edges_strat(G,k, num_interactions, matrices, delta=0, noise=0):
  edges=nx.edges(G)
  selected=random.sample(nx.edges(G), num_interactions)
  #choose k edges among the ones in the graph
  for pair in selected:
    v=pair[0]
    w=pair[1]
    action_v=action(G, intention(G,v), noise)
    action_w=action(G, intention(G,w), noise)
    G.node[v]['payoffs'].append(payoff_mtx[action_v][action_w][0])
    G.node[w]['payoffs'].append(payoff_mtx[action_v][action_w][1])
  for v in nx.nodes(G):
    if len(G.node[v]['payoffs']) != 0:
      G.node[v]['fitness']=normalization(Theta_paper1(sum(G.node[v]['payoffs'])/len(G.node[v]['payoffs']), delta))
    #restart payoff list for next round
    G.node[v]['payoffs']=[]
  return

def interaction_some_edges_fitness(G,k, num_interactions, matrices, delta=0, noise=0):
  edges=nx.edges(G)
  selected=random.sample(nx.edges(G), num_interactions)
  #choose k edges among the ones in the graph
  for pair in sleected:
    a=pair[0]
    b=pair[1]
    if random.random()<function(G.node[b]['fitness']):
      D[(a,b)]='Defect'
    else:
      D[(a,b)]='Cooperate'
    if random.random()<function(G.node[a]['fitness']):
      D[(b,a)]='Defect'
    else:
      D[(b,a)]='Cooperate'
  for a in nx.nodes(G):
    for b in G.neighbors(a):
      action_a=action(G, D[(a,b)], noise)
      action_b=action(G, D[(b,a)], noise)
      G.node[a]['payoffs'].append(payoff_mtx[action_a][action_b][0])
      G.node[b]['payoffs'].append(payoff_mtx[action_a][action_b][1])
  for v in nx.nodes(G):
    if len(G.node[v]['payoffs']) != 0:
      G.node[v]['fitness']=normalization(Theta_paper1(sum(G.node[v]['payoffs'])/len(G.node[v]['payoffs']), delta))
      G.node[v]['payoffs']=[]
  return G

'''-------------------
GRAPH FOR TESTING
-------------------'''

#strat_list=['Cooperate', 'Defect']
#G=init.generate_graph([3,4], 'triangular_lattice')
#init.label_birth_death(G, strat_list)
#init.label_utkovski(G)


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








          


