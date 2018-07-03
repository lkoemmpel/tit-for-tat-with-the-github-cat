'''-------------------
    PACKAGES
-------------------'''

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sys

'''
INPUTS: 
  Number of nodes
  Number of cliques
  Type of graph

OUTPUTS:
  Some form of a graph
'''

def generate_lattice(n, m, type = 'triangular', dim = 2, periodic = False, with_positions = True, create_using = None):
    try:
        if type == 'triangular':
            lattice = nx.triangular_lattice_graph(10, 20, periodic=False, create_using=None)
            #nx.draw(lattice)
            #plt.show()
            return lattice
    except ValueError:
        print("The specified lattice type was invalid.")



def generate_graph(n, d, type = 'random', periodic=False, with_positions=True, create_using=None):
  '''
  n = number of nodes
  d = average degree of the graph
  '''
  try:
    if type == 'hypercube':
      graph = nx.hypercube_graph(n)
      return graph
    else:
      if type == 'random':
        graph = nx.random_regular_graph(d, n)
        return graph


  except ValueError:
    print("The specified graph type was invalid.")

#print(generate_lattice(3,4))

def label_birth_death(G, strat_list):
  for n in nx.nodes(G):
    G.node[n]['strategy']=random.choice(strat_list)
    G.node[n]['fitness'] = random.uniform(0,1)
    G.node[n]['payoffs'] = []

def label_utkovski(G):
  for n in nx.nodes(G):
    G.node[n]['turn-payoff']=0
    G.node[n]['total-payoff']=0
    #cooperative_state = probability of helping another node
    G.node[n]['coop-state'] = random.uniform(0,1)


