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



def generate_graph(n, type = 'random', d=0, m=0, periodic=False, with_positions=True, create_using=None):
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
    elif type == 'complete':
      graph = nx.complete_graph(n)
    elif type == 'dumbbell':
      graph=nx.barbell_graph(n//2-1, n-2*(n//2-1))
    return graph 



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


def label_birth_death(G, strat_list, start_prop_coop=None):
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
    if start_prop_coop != None:
      if random.uniform(0,1) <= start_prop_coop:
        G.node[n]['strategy']= 'Cooperate'
      else:
        strat_list.remove('Cooperate')
        G.node[n]['strategy'] = random.choice(strat_list)
        strat_list.append('Cooperate')
    else:
        G.node[n]['strategy'] = random.choice(strat_list)

    #G.node[n]['fitness'] = random.uniform(0,1)
    G.node[n]['fitness'] = 1
    G.node[n]['payoffs'] = []

def label_dumbbell_birth_death(G, strat_list, prop_coop_left=1, prop_coop_right=0):
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
  first_node = True
  connecting_nodes = []
  for n in nx.nodes(G):
    if G.degree[n] > 2:
      #this is a node in one of the cliques
      if first_node:
        G.node[n]['strategy'] = 'Cooperate'
        first_node = False
      else:
        labeled = False
        for neighbor in G.neighbors(n):
          try:
            if G.node[neighbor]['strategy'] == 'Cooperate':
              #We're on the cooperate end of the dumbbell
              G.node[n]['strategy'] = 'Cooperate'
              labeled = True
            if G.node[neighbor]['strategy'] == 'Defect':
              #We're on the defect end of the dumbbell
              G.node[n]['strategy'] = 'Defect'
              labeled = True
          except KeyError:
              #no conclusive evidence from neighbors
              pass
        if not labeled:
          #The node was not determined to be a cooperator or defector because of it's neighbors
          # The defector end may have no labels yet, but the cooperator end has at least one
          # so the node can't be on the cooperator end.
          G.node[n]['strategy'] = 'Defect'

    else:
      # We're at one of the connecting nodes
      connecting_nodes.append(n)

    G.node[n]['fitness'] = 1
    G.node[n]['payoffs'] = []


    #Now we go back and label the connecting nodes
    for n in connecting_nodes:
      G.node[n]['strategy'] = random.choice(strat_list)


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
    if n[0] == width//2:
    #if n[0] == math.floor(width/2):
      # this node is along the dimension we want 
      G.node[n]['strategy']= 'Defect'
    else:
      G.node[n]['strategy']= 'Cooperate'

    #G.node[n]['fitness'] = random.uniform(0,1)
    G.node[n]['fitness'] = 0.5
    G.node[n]['payoffs'] = []

def label_utkovski(G):
  '''
    INPUTS: 
    G               The graph
  
    OUTPUTS:
    None, but labels graph, so for each node:
        initiates 'turn-payoff' at 0
        initiates 'total-payoff' at 0
        assigns a random number from 0 to 1 as 'fitness'
  '''
  for n in nx.nodes(G):
    G.node[n]['turn-payoff']=0
    G.node[n]['total-payoff']=0
    #cooperative_state = probability of helping another node
    G.node[n]['coop-state'] = random.uniform(0,1)


