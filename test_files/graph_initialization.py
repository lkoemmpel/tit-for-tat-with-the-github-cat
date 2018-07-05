'''-------------------
    PACKAGES
-------------------'''

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

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
        if dim == 2:
          if type == 'triangular':
              lattice = nx.triangular_lattice_graph(n, m, periodic=False, create_using=None)
              #nx.draw(lattice)
              #plt.show()
              return lattice
    except ValueError:
        print("The specified lattice type was invalid.")



def generate_graph(n, d, m=0, type = 'random', periodic=False, with_positions=True, create_using=None):
  '''
  n = number of nodes
  d = average degree of the graph
  '''
  try:
    if type == 'hypercube':
      graph = nx.hypercube_graph(n)
      return graph
    elif type == 'random':
        graph = nx.random_regular_graph(d, n)
        return graph
    elif type == 'erdos_renyi':
        graph=nx.erdos_renyi_graph(n, m)


  except ValueError:
    print("The specified graph type was invalid.")

def generate_weighted(n, d, m=0, type = 'random', periodic=False, with_positions=True, create_using=None):
  '''
  n = number of nodes
  d = average degree of the graph
  '''
  try:
    if type == 'hypercube':
      graph = nx.hypercube_graph(n)
      return graph
    elif type == 'random':
        graph = nx.random_regular_graph(d, n)
        return graph
    elif type == 'erdos_renyi':
        graph=nx.erdos_renyi_graph(n, m)
    weights={}
    for edge in graph.edges:
      weights[edge]=random.random()
    nx.set_edge_attributes(graph, 'weight', weights)


  except ValueError:
    print("The specified graph type was invalid.")


def label_birth_death(G, strat_list):
  for n in nx.nodes(G):
    G.node[n]['strategy']=random.choice(strat_list)
    G.node[n]['fitness'] = random.uniform(0,1)
    G.node[n]['payoffs'] = []



def label_BD_according_to_one_dim(G, strat_list, width):
  '''
  ---------------TODO--------------
  Edit to accomodate more strategies
  '''
  for n in nx.nodes(G):
    if n[0] == math.floor(width/2):
      # this node is along the dimension we want 
      G.node[n]['strategy']= 'Defect'
    else:
      G.node[n]['strategy']= 'Cooperate'

    G.node[n]['fitness'] = random.uniform(0,1)
    G.node[n]['payoffs'] = []




def label_utkovski(G):
  for n in nx.nodes(G):
    G.node[n]['turn-payoff']=0
    G.node[n]['total-payoff']=0
    #cooperative_state = probability of helping another node
    G.node[n]['coop-state'] = random.uniform(0,1)


