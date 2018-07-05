'''-------------------
    PACKAGES
-------------------'''

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

def generate_lattice(n, m, type = 'triangular', dim = 2, periodic = False, with_positions = True, create_using = None):
    '''
    INPUTS: 
    n               Number of nodes horizontally
    m               Number of nodes vertically
    type            Type of graph
    dim             Dimension
    periodic        Bool: is the graph periodic?
    with_positions
    create_using

    OUTPUTS:
    Lattice triangular graph with the specified parameters
    '''
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
    INPUTS: 
    n               Number of nodes 
    d               Average degree of the graph
    type            Type of graph
    periodic        Bool: is the graph periodic?
    with_positions
    create_using

    OUTPUTS:
    Graph with the specified parameters and of the specified type
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
    INPUTS: 
    n               Number of nodes 
    d               Average degree of the graph
    m               Number of total edges in the graph
    type            Type of graph
    periodic        Bool: is the graph periodic?
    with_positions
    create_using

    OUTPUTS:
    Graph with the specified parameters and of the specified type, 
    and with randomly (0 to 1) assigned edge weights
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
  '''
    INPUTS: 
    G               The graph
    strat_list      List containing the strategy labels/strings

    OUTPUTS:
    None, but modifies graph: 
        assigns Cooperate/Defect with prob 1/2 each
        every node has some value from 0 to 1 as fitness
        for every node, a turn payoff list is introduced
  '''
  for n in nx.nodes(G):
    G.node[n]['strategy']=random.choice(strat_list)
    G.node[n]['fitness'] = random.uniform(0,1)
    G.node[n]['payoffs'] = []

def label_BD_according_to_one_dim(G, strat_list, width):
  '''
    INPUTS: 
    G               The graph
    strat_list      List containing the strategy labels/strings
    width           width of the graph in 1st dimension

    OUTPUTS:
    None, but labels graph:
        assigns 'Defect' only to the nodes with the first coord width//2
        every node has some value from 0 to 1 as fitness
        for every node, a turn payoff list is introduced  
  ---------------TODO--------------
  Edit to accomodate more strategies
  '''
  for n in nx.nodes(G):
    #if n[0] == width//2:
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


