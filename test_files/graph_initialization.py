'''-------------------
    PACKAGES
-------------------'''

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import itertools


'''---------------------
    GRAPH GENERATORS
---------------------'''

def generate_graph(parameters, type = 'random'):

  #n, type='random', d=0, m=0, k=5, p=.5, periodic=False, with_positions=True, create_using=None
  '''

    INPUTS: 
    
    type              Type of graph
    parameters        List of parameters specific to the type of graph

    OUTPUTS:
    Graph satisfying the specified parameters and of the specified type
  '''
  try:
    if type == 'triangular_lattice':
      n_dim = parameters[0]
      m_dim = parameters[1]
      graph = nx.triangular_lattice_graph(n_dim, m_dim)
      return graph

    if type == 'hypercube':
      num_nodes=parameters[0]
      graph = nx.hypercube_graph(num_nodes)
      return graph

    elif type == 'random':
      num_nodes=parameters[0]
      av_deg=parameters[1]
      graph = nx.random_regular_graph(av_deg, num_nodes)
      return graph

    elif type == 'erdos_renyi':
        num_nodes=parameters[0]
        edges=parameters[1]
        graph=nx.erdos_renyi_graph(num_nodes, edges)

    elif type == 'complete':
      num_nodes=parameters[0]
      graph = nx.complete_graph(num_nodes)

    elif type == 'dumbell':
      n=parameters[0]
      graph=nx.barbell_graph(n//2-1, n-2*(n//2-1))

    elif type == 'complete_bipartite':
      m=parameters[0]
      n=parameters[1]
      graph=nx.complete_bipartite_graph(m,n)

    elif type == 'dumbell_multiple':
      size_dumbell=parameters[0]
      num_dumbell=parameters[1]
      size_path=parameters[2]
      graph=generate_dumbell_multiple_cliques(size_dumbell, num_dumbell, size_path)

    elif type == 'rich_club':
      size_club = parameters[0]
      size_periphery = parameters[1]
      #prob_rp = parameters[2]
      #prob_rr = parameters[3]
      #prob_pp = parameters[4]
      num_peripheries=parameters[2]
      a = parameters[3]
      b = parameters[4]
      c = parameters[5]
      graph=generate_rich_club_adapted_version(size_club, size_periphery, num_peripheries, a, b, c)

    elif type == 'dumbell_multiple_sized':
      size_list=parameters[0]
      path_length=parameters[1]
      graph=generate_dumbell_multiple_sizes(size_list, path_length)

    elif type == 'dumbell_string':
      sizes=parameters[0]
      lengths=parameters[1]
      graph=generate_dumbell_string(sizes,lengths)

    elif type == 'with_indicator':
      indicator=parameters[0]
      graph=generate_dumbell_indicator_connectionstrength(indicator)

    return graph 

  except ValueError:
    print("The specified graph type was invalid.")

def generate_weighted(parameters, type = 'random'):

  #n, type='random', d=0, m=0, k=5, p=.5, periodic=False, with_positions=True, create_using=None
  '''

    INPUTS: 
    
    type              Type of graph
    parameters        List of parameters specific to the type of graph

    OUTPUTS:
    Graph satisfying the specified parameters and of the specified type
  '''
  try:
    if type == 'triangular_lattice':
      n_dim = parameters[0]
      m_dim = parameters[1]
      graph = nx.triangular_lattice_graph(n_dim, m_dim)
      return graph

    if type == 'hypercube':
      num_nodes=parameters[0]
      graph = nx.hypercube_graph(num_nodes)
      return graph

    elif type == 'random':
      num_nodes=parameters[0]
      av_deg=parameters[1]
      graph = nx.random_regular_graph(av_deg, num_nodes)
      return graph

    elif type == 'erdos_renyi':
        num_nodes=parameters[0]
        edges=parameters[1]
        graph=nx.erdos_renyi_graph(num_nodes, edges)

    elif type == 'complete':
      num_nodes=parameters[0]
      graph = nx.complete_graph(num_nodes)

    elif type == 'dumbell':
      n=parameters[0]
      graph=nx.barbell_graph(n//2-1, n-2*(n//2-1))

    elif type == 'complete_bipartite':
      m=parameters[0]
      n=parameters[1]
      graph=nx.complete_bipartite_graph(m,n)

    elif type == 'dumbell_multiple':
      size_dumbell=parameters[0]
      num_dumbell=parameters[1]
      size_path=parameters[2]
      graph=generate_dumbell_multiple_cliques(size_dumbell, num_dumbell, size_path)

    elif type == 'rich_club':
      size_club = parameters[0]
      size_periphery = parameters[1]
      #prob_rp = parameters[2]
      #prob_rr = parameters[3]
      #prob_pp = parameters[4]
      num_peripheries=parameters[2]
      a = parameters[3]
      b = parameters[4]
      c = parameters[5]
      graph=generate_rich_club_adapted_version(size_club, size_periphery, num_peripheries, a, b, c)

    elif type == 'dumbell_multiple_sized':
      size_list=parameters[0]
      path_length=parameters[1]
      graph=generate_dumbell_multiple_sizes(size_list, path_length)

    elif type == 'dumbell_string':
      sizes=parameters[0]
      lengths=parameters[1]
      graph=generate_dumbell_string(sizes,lengths)

    elif type == 'with_indicator':
      indicator=parameters[0]
      graph=generate_dumbell_indicator_connectionstrength(indicator)


    if graph:
      weights={}
      for edge in graph.edges():
        weights[edge]=1
      nx.set_edge_attributes(graph, weights, 'weight')
      return graph

    return graph 

  except ValueError:
    print("The specified graph type was invalid.")

def generate_graph_original(n, type = 'random', d=0, m=0, k=5, p=.5, periodic=False, with_positions=True, create_using=None):
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
    elif type == 'dumbell':
      graph=nx.barbell_graph(n//2-1, n-2*(n//2-1))
    elif type == 'dumbell_multiple':
      m=10
      N=5
      L=2
      graph=generate_dumbell_multiple_cliques(m, N, L)
    return graph 



  except ValueError:
    print("The specified graph type was invalid.")

def generate_dumbell_multiple_cliques(m, N, L):
  '''
  INPUTS:
  m       number of nodes in each clique
  N       number of cliques
  L       number of nodes in a connecting path
  prop_coop the proportion of cooperators wanted, used in
        stochastic labeling as a probability

  OUTPUTS:
  a graph satisfying all of the above parameters, with mN+L*\choose{m}{2}
  '''
  if m<2:
      raise nx.NetworkXError(\
            "Invalid graph description, n should be >=2")
  if L<0:
      raise nx.NetworkXError(\
            "Invalid graph description, L should be >=0")
  edges = []
  G = nx.Graph()

  #N times
  for k in range(N):
    #add a clique that contains the nodes:
    #     km+1, km+2, ... , (k+1)m+1  
    range_clique=range(k*m+1,(k+1)*m+1)
    for pair in itertools.combinations(range_clique,2):
      a,b = pair[0], pair[1]
      edges.append( (( (k,k) ,a),( (k,k) ,b)) )
    #edges+= [pair for pair in itertools.combinations(range_clique,2)]

  for pair in itertools.combinations(range(N),2):
    #here we are focusing on connecting the nodes 
    #     an+1  and   bn+1
    #where the unordered pair that we are focusing on is (a,b)
    a=pair[0]
    b=pair[1]
    if L>1:
      edges.append( ( ((a,a),a*m+1),(pair,1) ) )
      edges.append( ((pair,L), ((b,b),b*m+1)) )
      for k in range(1,L):
        edges.append( ((pair,k),(pair,k+1)) )
    elif L==1:
      edges.append( (((a,a),a*m+1),(pair,1)) )
      edges.append( ((pair,1),((b,b),b*m+1)) )
    else:
      edges.append( (((a,a),a*m+1), ((b,b),b*m+1) )) 
  G.add_edges_from(edges)

  for n in G.nodes():
    G.nodes[n]['coord']=n
  G=nx.convert_node_labels_to_integers(G)
  return G

def generate_dumbell_multiple_sizes(sizes, path_length):
  nodes_so_far=0
  sums={}
  edges=[]
  G=nx.Graph()
  for clique_index in range(len(sizes)):
    prev=nodes_so_far
    nodes_so_far+=sizes[clique_index]
    #later going to be used for the paths
    sums[clique_index]=prev

    range_clique=range(prev+1, nodes_so_far+1)
    for pair in itertools.combinations(range_clique,2):
      a,b = pair[0], pair[1]
      edges.append( (( (clique_index,clique_index) ,a),( (clique_index,clique_index) ,b)) )
  #NOW PATH EDGES!
  for pair in itertools.combinations(range(len(sizes)),2):
    a = pair[0]
    b = pair[1]
    if path_length>1:
      edges.append( (((a,a),sums[a]+1),(pair,1)) )
      edges.append( ((pair,path_length), ((b,b),sums[b]+1)) )
      for k in range(1,path_length):
        edges.append( ((pair,k),(pair,k+1)) )
    elif path_length==1:
      edges.append( (((a,a),sums[a]+1),(pair,1)) )
      edges.append( ((pair,1),((b,b),sums[b]+1)) )
    else:
      edges.append(((a,a),sums[a]+1), ((b,b),sums[b]+1))
  G.add_edges_from(edges)

  for n in G.nodes():
    G.nodes[n]['coord']=n
  G=nx.convert_node_labels_to_integers(G)

  #for n in G.nodes():
  #  print(G.nodes[n]['coord'])
  return G

def generate_dumbell_string(sizes, lengths):
  nodes_so_far=0
  sums={}
  edges=[]
  G=nx.Graph()
  for clique_index in range(len(sizes)):
    prev=nodes_so_far
    nodes_so_far+=sizes[clique_index]
    #later going to be used for the paths
    sums[clique_index]=prev

    range_clique=range(prev+1, nodes_so_far+1)
    for pair in itertools.combinations(range_clique,2):
      a,b = pair[0], pair[1]
      edges.append( (( (clique_index,clique_index) ,a),( (clique_index,clique_index) ,b)) )
  #NOW PATH EDGES!
  list_adjs=[(i,i+1) for i in range(len(sizes)-1)]
  for index in range(len(list_adjs)):
    a = list_adjs[index][0]
    b = list_adjs[index][1]
    pair=(a,b)
    path_length=lengths[index]
    if path_length>1:
      edges.append( (((a,a),sums[a]+1),(pair,1)) )
      edges.append( ((pair,path_length), ((b,b),sums[b]+1)) )
      for k in range(1,path_length):
        edges.append( ((pair,k),(pair,k+1)) )
    elif path_length==1:
      edges.append( (((a,a),sums[a]+1),(pair,1)) )
      edges.append( ((pair,1),((b,b),sums[b]+1)) )
    else:
      edges.append(((a,a),sums[a]+1), ((b,b),sums[b]+1))
  G.add_edges_from(edges)

  for n in G.nodes():
    G.nodes[n]['coord']=n
  G=nx.convert_node_labels_to_integers(G)

  #for n in G.nodess():
  #  print(G.nodes[n]['coord'])
  return G

def generate_rich_club(size_club, size_periphery, prob_rp=1, prob_rr=1, prob_pp=0):
  graph=nx.complete_graph(size_club)
  for A in range(size_club):
    for B in range(size_club+1,size_club+size_periphery+1):
      #add edge with Pr[rich<->poor]
      if random.random()<prob_rp:
        graph.add_edge(A,B)
  for r_1 in range(size_club):
    for r_2 in range(size_club):
      m , M = min(r_1,r_2), max(r_1,r_2)
      #add edge with Pr[rich<->rich]
      if random.random()<prob_rr:
        graph.add_edge(m,M)
  for p_1 in range(size_club+1,size_club+size_periphery+1):
    for p_2 in range(size_club+1,size_club+size_periphery+1):
      m, M = min(p_1,p_2), max(p_1,p_2)
      #add edge with Pr[poor<->poor]
      if random.random()<prob_pp:
        graph.add_edge(m,M)
  return graph

def generate_rich_club_adapted_version(size_club, size_periphery, num_peripheries, a, b, c):
  indicator=nx.Graph()
  sizes={0:size_club}
  strengths={}
  for index in range(num_peripheries):
    sizes[index+1]=size_periphery
    strengths[(0,index+1)]=b
  #MAING OF INDICATOR GRAPH
  indicator.add_edges_from(strengths.keys())
  nx.set_node_attributes(indicator, name='size', values=sizes)
  nx.set_edge_attributes(indicator, name='strength', values=strengths)
  graph=generate_dumbell_indicator_connectionstrength(indicator)
  #GETTING CLIQUE RANGES:
  cliques_to_sets={x:set() for x in range(num_peripheries+1)}
  #put every node in a box, labeled with the number of clique that the node is in
  for n in nx.nodes(graph):
    k = graph.nodes[n]['coord'][0][0]
    cliques_to_sets[k].add(n)
  #for every clique
  for k in cliques_to_sets.keys():
    clique_set = cliques_to_sets[k]
    visited=set()
    for A in clique_set:
      for B in clique_set:
        #Thre conditionals: not visited yet, the two different
        if A!=B and (A,B) not in visited:
            visited.add((A,B))
            visited.add((B,A))
            m, M = min(A,B), max(A,B)
            #new=((k,k),m), ((k,k),M)
            new=(m,M)
            L=[new]
            if k == 0:
              D = {new : a}
            else:
              D = {new : c}
            graph.add_edges_from(L)
            nx.set_edge_attributes(graph, D, 'weight')
  return graph

def generate_dumbell_indicator_pathlength(indicator):
  nodes_so_far=0
  sums={}
  edges=[]
  G=nx.Graph()
  index_to_range={}
  for clique_index in range(len(nx.nodes(indicator))):
    prev=nodes_so_far
    nodes_so_far += indicator.nodes[clique_index]['size']
    #later going to be used for the paths
    sums[clique_index]=prev

    range_clique=range(prev+1, nodes_so_far+1)
    index_to_range[clique_index]=list(range_clique)
    for pair in itertools.combinations(range_clique,2):
      a,b = pair[0], pair[1]
      edges.append( (( (clique_index,clique_index) ,a),( (clique_index,clique_index) ,b)) )
  #print(index_to_range)
  #NOW PATH EDGES!
  lengths=nx.get_edge_attributes(indicator, 'length')
  for pair in nx.edges(indicator):
    #for index in range(len(list_adjs)):
    a = pair[0]
    b = pair[1]
    #path_length=indicator.edge[pair]['length']
    path_length=lengths[pair]
    #CONNECTORS:
    connector_a=random.choice(index_to_range[a])
    connector_b=random.choice(index_to_range[b])
    if path_length>1:
      edges.append( (((a,a),connector_a),(pair,1)) )
      edges.append( ((pair,path_length), ((b,b),connector_b)) )
      for k in range(1,path_length):
        edges.append( ((pair,k),(pair,k+1)) )
    elif path_length==1:
      edges.append( (((a,a),connector_a),(pair,1)) )
      edges.append( ((pair,1),((b,b),connector_b)) )
    else:
      edges.append(((a,a),connector_a), ((b,b),connector_b))
  G.add_edges_from(edges)

  for n in G.nodes():
    G.nodes[n]['coord']=n
  G=nx.convert_node_labels_to_integers(G)

  #for n in G.nodes():
  #  print(G.nodes[n]['coord'])
  return G

def generate_dumbell_indicator_connectionstrength(indicator):
  nodes_so_far=0
  sums={}
  edges=[]
  G=nx.Graph()
  index_to_range={}
  for clique_index in range(len(nx.nodes(indicator))):
    prev=nodes_so_far
    nodes_so_far += indicator.nodes[clique_index]['size']
    #later going to be used for the paths
    sums[clique_index]=prev

    range_clique=range(prev+1, nodes_so_far+1)
    index_to_range[clique_index]=list(range_clique)
    for pair in itertools.combinations(range_clique,2):
      a,b = pair[0], pair[1]
      edges.append( (( (clique_index,clique_index) ,a),( (clique_index,clique_index) ,b)) )
  #add all of the edges produced so far!
  G.add_edges_from(edges)
  strength_dictio={edge: 1 for edge in edges}
  nx.set_edge_attributes(G, strength_dictio, 'weight')
  #NOW CONECTIONS BETWEEN PAIRS OF DUMBELLS!
  strengths=nx.get_edge_attributes(indicator, 'strength')
  for pair in nx.edges(indicator):
    #for index in range(len(list_adjs)):
    a = pair[0]
    b = pair[1]
    #path_length=indicator.edge[pair]['length']
    connection_strength=strengths[pair]
    #CONNECTEVERY PAIR:
    for local_in_a in index_to_range[a]:
      for local_in_b in index_to_range[b]:
        new=((a,a),local_in_a), ((b,b),local_in_b)
        L=[new]
        D={new:strengths[pair]}
        G.add_edges_from(L)
        nx.set_edge_attributes(G, D, 'weight')

  for n in G.nodes():
    G.nodes[n]['coord']=n
  G=nx.convert_node_labels_to_integers(G)

  #for n in G.nodes():
  #  print(G.nodes[n]['coord'])
  return G

'''---------------------
    GRAPH LABELING
---------------------'''

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
        G.nodes[n]['strategy']= 'Cooperate'
      else:
        strat_list.remove('Cooperate')
        G.nodes[n]['strategy'] = random.choice(strat_list)
        strat_list.append('Cooperate')
    else:
        G.nodes[n]['strategy'] = random.choice(strat_list)

    G.nodes[n]['fitness'] = random.uniform(0,1)
    G.nodes[n]['payoffs'] = []

def label_allen(G,b,c,strat_list, start_prop_coop=None):
  #SET STRATEGIES and s_i
  num_nodes=len(G.nodes())
  num_coops=int(round(num_nodes*start_prop_coop))
  selected_coops=random.sample(list(G.nodes()), num_coops)
  selected_coops=set(selected_coops)
  for n in G.nodes():
    if n in selected_coops:
      G.nodes[n]['strategy'] = 'Cooperate'
      G.nodes[n]['s']=1
    else:
      G.nodes[n]['strategy'] = 'Defect'
      G.nodes[n]['s']=0


  #SET f_i VALUES
  for i in G.nodes():

    G.nodes[i]['w'] = w_i(G, i)

    G.nodes[i]['f0'] = -c*G.nodes[i]['s']
    for j in G.neighbors(i):
      G.nodes[i]['f0'] += b*prob_n_step_walk(G,i,j,1)*G.nodes[j]['s']

    G.nodes[i]['f2'] = -c*G.nodes[i]['s']
    for j in G.neighbors(i):
      G.nodes[i]['f2'] += b*prob_n_step_walk(G,i,j,2)*G.nodes[j]['s']

    G.nodes[i]['F'] = 1+delta*G.nodes[i]['f0']
    G.nodes[i]['pi'] = reproductive_value(G,i)
    G.nodes[i]['payoffs'] = []
    G.nodes[i]['fitness'] = random.uniform(0,1)

def label_birth_death_precise_prop(G,strat_list, start_prop_coop=None):
  num_nodes=len(G.nodes())
  num_coops=int(round(num_nodes*start_prop_coop))
  selected_coops=random.sample(list(G.nodes()), num_coops)
  selected_coops=set(selected_coops)
  for n in G.nodes():
    if n in selected_coops:
      G.nodes[n]['strategy'] = 'Cooperate'
    else:
      G.nodes[n]['strategy'] = 'Defect'
    G.nodes[n]['fitness'] = random.uniform(0,1)
    G.nodes[n]['payoffs'] = []

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
      
      print("\n")
      print("We have found a node in a clique labeled ", n)
      
      #this is a node in one of the cliques
      if first_node:
        
        print("This node is the first clique node; we assign strategy cooperate.")
        
        G.nodes[n]['strategy'] = 'Cooperate'
        first_node = False
      else:
        print("This node is not the first clique node")
        labeled = False
        for neighbor in G.neighbors(n):
          try:
            if G.nodes[neighbor]['strategy'] == 'Cooperate':
              #We're on the cooperate end of the dumbbell
              
              print("------We found a neighbor ", neighbor, " with a cooperative strategy")
              print("Assigning strategy cooperate to node ", n)
              
              G.nodes[n]['strategy'] = 'Cooperate'
              labeled = True
              break
            if G.nodes[neighbor]['strategy'] == 'Defect':
              #We're on the defect end of the dumbbell
              
              print("------We found a neighbor ", neighbor, " with a defect strategy")
              print("Assigning strategy defect to node ", n)
              
              G.nodes[n]['strategy'] = 'Defect'
              labeled = True
              break
          except KeyError:
            print("---------this neighbor, neighbor ", neighbor, " had no strategy label")
            #no conclusive evidence from neighbors
            pass
        if not labeled:
          print("No labeled neighbors, so we label node ", n, " as defect")
          #The node was not determined to be a cooperator or defector because of it's neighbors
          # The defector end may have no labels yet, but the cooperator end has at least one
          # so the node can't be on the cooperator end.
          G.nodes[n]['strategy'] = 'Defect'

    else:
      # We're at one of the connecting nodes
      print("We found a connecting node labeled ", n)
      connecting_nodes.append(n)
    
    print("Assigning fitness and payoffs")
    
    G.nodes[n]['fitness'] = 1
    G.nodes[n]['payoffs'] = []


  #Now we go back and label the connecting nodes
  for c in connecting_nodes:
    
    print("Labeling connecting node ", c)
    
    G.nodes[c]['strategy'] = random.choice(strat_list)   

def label_dumbell_multiple_cliques(G, strat_list, clique_to_prop):
  '''
  G                   A multiple dumbell graph
  clique_to_prop      dictionary maps clique_index---->prop of cooperators    
  '''
  sizes={}
  num_cliques=0
  for n in G.nodes():
    #if this is a clique node
    if G.nodes[n]['coord'][0][0] == G.nodes[n]['coord'][0][1]:
      clique_num = G.nodes[n]['coord'][0][0]
      num_cliques = max(num_cliques, clique_num)
      try:
        sizes[clique_num] += 1
      except:
        sizes[clique_num] = 1
  
  sets={clique_num:set() for clique_num in range(num_cliques+1)}
  for n in G.nodes():
    #if this is a clique node
    if G.nodes[n]['coord'][0][0] == G.nodes[n]['coord'][0][1]:
      clique_num = G.nodes[n]['coord'][0][0]
      sets[clique_num].add(n)
  chosen = {}
  for index in sizes.keys():
    prop = clique_to_prop[index]
    num_coop = round(prop*sizes[index])
    chosen[index] = random.sample(sets[index], num_coop)

  for n in G.nodes():
    #if this is a clique node
    if G.nodes[n]['coord'][0][0] == G.nodes[n]['coord'][0][1]:
      clique_num = G.nodes[n]['coord'][0][0]
      if n in chosen[clique_num]:
        G.nodes[n]['strategy'] = 'Cooperate'
        G.nodes[n]['s'] = 1
      else:
        G.nodes[n]['strategy'] = 'Defect'
        G.nodes[n]['s'] = 0
    else:
      G.nodes[n]['strategy'] = random.choice(strat_list)
      if G.nodes[n]['strategy'] == 'Defect':
        G.nodes[n]['s'] = 0
      elif G.nodes[n]['strategy'] == 'Cooperate':
        G.nodes[n]['s'] = 1

    '''
      if random.uniform(0,1)<clique_to_prop[clique_num]:
        G.nodes[n]['strategy'] = 'Cooperate'
      else:
        G.nodes[n]['strategy'] = 'Defect'
    else:
      G.nodes[n]['strategy'] = random.choice(strat_list)
    '''
    G.nodes[n]['fitness'] = random.uniform(0,1)
    G.nodes[n]['payoffs'] = []

def label_dumbell_multiple_cliques_allen(G, b, c, strat_list, clique_to_prop):
  '''
  G                   A multiple dumbell graph
  clique_to_prop      dictionary maps clique_index---->prop of cooperators    
  '''
  sizes={}
  num_cliques=0
  for n in G.nodes():
    #if this is a clique node
    if G.nodes[n]['coord'][0][0] == G.nodes[n]['coord'][0][1]:
      clique_num = G.nodes[n]['coord'][0][0]
      num_cliques = max(num_cliques, clique_num)
      try:
        sizes[clique_num] += 1
      except:
        sizes[clique_num] = 1
  
  sets={clique_num:set() for clique_num in range(num_cliques+1)}
  for n in G.nodes():
    #if this is a clique node
    if G.nodes[n]['coord'][0][0] == G.nodes[n]['coord'][0][1]:
      clique_num = G.nodes[n]['coord'][0][0]
      sets[clique_num].add(n)

  chosen = {}
  for index in sizes.keys():
    prop = clique_to_prop[index]
    num_coop = round(prop*sizes[index])
    chosen[index] = random.sample(sets[index], num_coop)

  for n in G.nodes():
    #if this is a clique node
    if G.nodes[n]['coord'][0][0] == G.nodes[n]['coord'][0][1]:
      clique_num = G.nodes[n]['coord'][0][0]
      if n in chosen[clique_num]:
        G.nodes[n]['strategy'] = 'Cooperate'
        G.nodes[n]['s'] = 1
      else:
        G.nodes[n]['strategy'] = 'Defect'
        G.nodes[n]['s'] = 0
    else:
      G.nodes[n]['strategy'] = random.choice(strat_list)
      if G.nodes[n]['strategy'] == 'Defect':
        G.nodes[n]['s'] = 0
      elif G.nodes[n]['strategy'] == 'Cooperate':
        G.nodes[n]['s'] = 1


  #SET f_i VALUES
  for i in G.nodes():

    G.nodes[i]['w'] = w_i(G, i)

    G.nodes[i]['f0'] = -c*G.nodes[i]['s']
    for j in G.neighbors(i):
      G.nodes[i]['f0'] += b*prob_n_step_walk(G,i,j,1)*G.nodes[j]['s']

    G.nodes[i]['f2'] = -c*G.nodes[i]['s']
    for j in G.neighbors(i):
      G.nodes[i]['f2'] += b*prob_n_step_walk(G,i,j,2)*G.nodes[j]['s']

    G.nodes[i]['F'] = 1+delta*G.nodes[i]['f0']
    G.nodes[i]['pi'] = reproductive_value(G,i)
    G.nodes[i]['payoffs'] = []
    G.nodes[i]['fitness'] = random.uniform(0,1)  

def label_rich_club_allen(G, b, c, strat_list, clique_to_prop):
  '''
  G                   A multiple dumbell graph
  clique_to_prop      dictionary maps clique_index---->prop of cooperators    
  '''
  for n in G.nodes():
    #if this is a clique node
    if G.nodes[n]['coord'][0][0] == G.nodes[n]['coord'][0][1]:
      clique_num = G.nodes[n]['coord'][0][0]
      if random.uniform(0,1)<clique_to_prop[clique_num]:
        G.nodes[n]['strategy'] = 'Cooperate'
        G.nodes[n]['s']=1
      else:
        G.nodes[n]['strategy'] = 'Defect'
        G.nodes[n]['s']=0
    else:
      G.nodes[n]['strategy'] = random.choice(strat_list)
      if G.nodes[n]['strategy']== 'Cooperate':
        G.nodes[n]['s']=1
      elif G.nodes[n]['strategy']== 'Defect':
        G.nodes[n]['s']=0
  #SET f_i VALUES
  for i in G.nodes():

    G.nodes[i]['w'] = w_i(G, i)

    G.nodes[i]['f0'] = -c*G.nodes[i]['s']
    for j in G.neighbors(i):
      G.nodes[i]['f0'] += b*prob_n_step_walk(G,i,j,1)*G.nodes[j]['s']

    G.nodes[i]['f2'] = -c*G.nodes[i]['s']
    for j in G.neighbors(i):
      G.nodes[i]['f2'] += b*prob_n_step_walk(G,i,j,2)*G.nodes[j]['s']

    G.nodes[i]['F'] = 1+delta*G.nodes[i]['f0']
    G.nodes[i]['pi'] = reproductive_value(G,i)
    G.nodes[i]['payoffs'] = []
    G.nodes[i]['fitness'] = random.uniform(0,1)  

def label_dumbell_multiple_cliques_precise(G, strat_list, clique_to_prop):
  '''
  G                   A multiple dumbell graph
  clique_to_prop      dictionary maps clique_index---->prop of cooperators    
  '''
  index_to_set={}
  selected_overall=set()
  num_cliques=0
  for n in G.nodes():
      if G.nodes[n]['coord'][0][0] == G.nodes[n]['coord'][0][1]:
        clique=G.nodes[n]['coord'][0][0]
        try:
          index_to_set[clique].add(n)
        except:
          index_to_set[clique] = {n}

  num_cliques = len(index_to_set)
  for k in range(num_cliques):
    num_elts = len(index_to_set[k])
    prop_coop = clique_to_prop[k]
    num_coops = int(round(num_elts*prop_coop))
    selected_clique = random.sample(index_to_set[k], num_coops)
    selected_overall = selected_overall.union(selected_clique)

  for n in G.nodes():
    #if this is a clique node
    if G.nodes[n]['coord'][0][0] == G.nodes[n]['coord'][0][1]:
      if n in selected_overall:
        G.nodes[n]['strategy'] = 'Cooperate'
      else:
        G.nodes[n]['strategy'] = 'Defect'
    else:
      G.nodes[n]['strategy'] = random.choice(strat_list)
    G.nodes[n]['fitness'] = random.uniform(0,1)
    G.nodes[n]['payoffs'] = []

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
      G.nodes[n]['strategy']= 'Defect'
    else:
      G.nodes[n]['strategy']= 'Cooperate'

    #G.nodes[n]['fitness'] = random.uniform(0,1)
    G.nodes[n]['fitness'] = 0.5
    G.nodes[n]['payoffs'] = []

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
    G.nodes[n]['turn_payoff']=0
    G.nodes[n]['total_payoff']=0
    #cooperative_state = probability of helping another node
    G.nodes[n]['coop_state'] = random.uniform(0,1)
    G.nodes[n]['strategy']=G.nodes[n]['coop_state']

def label_more_strategies(G, strat_list, dist_prob_strats=None):
  for n in nx.nodes(G):
    #-----------------
    #LABEL FOR STRATEGY
    #-----------------
    if dist_prob_strats:
      #dist_prob_strats is a dictionary that maps
      #strategy----> Pr [chosen by a node] ; sum values is 1
      cutoff=random.random()
      prev=0
      sum_now=0
      for strat in strat_list:
        prev=sum_now
        sum_now+=dist_prob_strats[strat]
        if prev<cutoff<sum_now:
          G.nodes[n]['strategy']=strat
    else:
      G.nodes[n]['strategy']=random.choice(strat_list)
    #--------------
    #LABEL FOR ASSESSMENT
    #--------------
    #memory
    G.nodes[n]['assessment']='good'
    G.nodes[n]['memory']=('helped','good')
    G.nodes[n]['fitness']=random.random()
    G.nodes[n]['payoffs']=[]


'''---------------------
    FROM DISPLAY.PY
---------------------'''

def color_and_draw_graph(G):
    '''
    INPUTS:     Graph with strategy node attributes
    OUTPUTS:    Graph where color maps has colors according to 
                node strategy attributes
    '''
    # initializes color map
    color_map = []
    for n in nx.nodes(G):
        if G.nodes[n]['strategy'] == 'Cooperate':
            color_map.append('green')
        else:
            color_map.append('red')

    # draws colored graph
    #plt.ion()
    nx.draw(G,node_color = color_map,with_labels = True)
    plt.show()
    #plt.pause(2.0)

    return G


'''---------------------------------------------------
    CALCULATIONS FROM THE PAPER 'ANY POP STRUCTURE'
----------------------------------------------------'''

delta=0.0005

def w_i(graph, i):
  w_i = 0
  for neighbor in graph.neighbors(i):
      w_i += graph[i][neighbor]['weight']
  return w_i

def prob_n_step_walk(graph, i, j, n):
    w_i = graph.nodes[i]['w']
    p_ij_sum = 0
    if n == 2:
        '''
        print("We've gotten to the n=2")
        '''
        #want to determine if there is a path from i to k to j
        for k in graph.neighbors(i):
            #there is a path from i to k
            if j in graph.neighbors(k):
                #there is a path from k to j
                p_ik = graph[i][k]['weight']/w_i
                p_kj = graph[k][j]['weight']/w_i
                p_ij_sum += p_ik * p_kj
    else:
        p_ij_sum = graph[i][j]['weight'] / w_i 
    return p_ij_sum

def reproductive_value(graph, i):
  w_i = graph.nodes[i]['w']
  W_sum = 0
  for j in graph.neighbors(i):
      W_sum += graph[i][j]['weight']
  return w_i/W_sum



'''---------------------
    TESTING GRAPHS
---------------------'''
graph={}
#---------------------
# graph[1]=generate_graph([10], 'hypercube')
# graph[2]=generate_graph([10, 5], 'random')
# graph[3]=generate_graph([10,30], 'erdos_renyi')
# graph[4]=generate_graph([20], 'complete')
# graph[5]=generate_graph([30],'dumbell')
# graph[6]=generate_graph([5,4,2], 'dumbell_multiple')
# graph[7]=generate_graph([10,30, .8,.999, .001],'rich_club')
# graph[8]=generate_graph([[5,7,9],2], 'dumbell_multiple_sized')
#---------------------
#sizes=[11,11,11,11]
#lengths=[2,2,5]
#graph[9]=generate_graph([sizes, lengths], 'dumbell_string')
#---------------------
#label_birth_death(graph[5], ['Cooperate','Defect'], 0.5)
#color_and_draw_graph(graph[5])
#---------------------
#label_birth_death(graph[3], ['Cooperate','Defect'], 0.5)
#color_and_draw_graph(graph[3])
#---------------------
#label_dumbell_multiple_cliques(graph[6], {0,3})
#color_and_draw_graph(graph[6])
#---------------------
#label_birth_death(graph[7], ['Cooperate','Defect'], 0.5)
#color_and_draw_graph(graph[7])
#---------------------
'''
indicator=nx.Graph()
sizes={0:10,
        1:10,
        2:10,
        3:10}

strengths={(0,1):0.2,
        (0,3):0.01,
        (1,2):0.01,
        (2,3):0.01}
#MAING OF INDICATOR GRAPH
indicator.add_edges_from(strengths.keys())
nx.set_node_attributes(indicator, name='size', values=sizes)
nx.set_edge_attributes(indicator, name='strength', values=strengths)
#GENERATE graph[10]
graph[10]=generate_graph([indicator], 'with_indicator')
clique_to_prop={0:0.5,1:0.9,2:0.9,3:0.9}
label_dumbell_multiple_cliques_precise(graph[10], ['Cooperate','Defect'], clique_to_prop)
#color_and_draw_graph(graph[10])
'''



'''
graph_type='rich_club'
parameters=[5, 30, 3, 1, 1, 0]
#GENERATE graph[10]
graph[11]=generate_graph(parameters, 'rich_club')
clique_to_prop={0:0.5,1:0.9,2:0.9,3:0.9}
label_dumbell_multiple_cliques_precise(graph[11], ['Cooperate','Defect'], clique_to_prop)
color_and_draw_graph(graph[11])
print(len(nx.nodes(graph[11])))
'''
