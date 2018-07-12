'''-------------------
        PACKAGES
-------------------'''

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import math


'''-------------------
        IMPORTS
-------------------'''
import graph_initialization as init



# add inherited fitness
# create probabilistic cooperative state from Utkovsky paper
# add mutation affecting reproduction
# replicate early results like pop density paper
# do work on 3d lattice




#----------------Can be uncommented for testing purposes--------------------------------
'''
# Initialization of graph to use for testing purposes
G = nx.triangular_lattice_graph(2,3)
for n in nx.nodes(G):
    # Assign node attributes
    # Initial strategy is C or D, chosen randomly
    strategy_indicator = random.randrange(0, 2, 1)
    if strategy_indicator == 1:
        G.node[n]['strategy'] = 'Cooperate'
        #color_map.append('blue')
    else:
        G.node[n]['strategy'] = 'Defect'
        #color_map.append('red')
    # fitness attributes are chosen randomly
    # print("The current strategy of node ", n, " is ", G.node[n]['strategy'] )
    G.node[n]['fitness'] = random.uniform(0,1)


strat_list = ['Cooperate', 'Defect', 'Tit-for-tat']
'''




'''------
REPRODUCTION
make all of the reproduction process based on the fitness, weight and mutation parameters.
--------'''

def birth_death(G, strat_list, u, num_rep):


    #CHOOSE REPRODUCING NODES
    fitness_dict = nx.get_node_attributes(G, 'fitness')
    #print("The fitness dictionary is ", fitness_dict)
    fitness_sum = sum(fitness_dict.values())
    #print("The fitness sum is ", fitness_sum)
    cutoffs=[]
    for num in range(num_rep):
        cutoffs.append(random.uniform(0, fitness_sum))
    #print("The cutoffs are "+ str(cutoffs))
    reproduced_nodes=set()
    for cut in cutoffs:
        last=0
        marker=0
        for n in nx.nodes(G):
            last=marker
            marker+=nx.get_node_attributes(G, 'fitness')[n]
            if last<cut<marker:
                if list(G.adj[n].keys()) != []:
                    reproduced_nodes.add(n)
                break
    old_strategies=[]
    reproduced_strategies=[]
    #print(reproduced_nodes)
    for i in reproduced_nodes:
        reproduced_strategy=G.node[i]['strategy']
        #set rep strat
        j = random.choice(list(G.adj[i].keys()))
        old_strategy = G.node[j]['strategy']

        difference=G.node[i]['fitness']-G.node[j]['fitness']
        if random.random()<=q(difference, .1, 3):
            #i will replace j
            #now decide which strategy will be inherited!
            mistake_indicator = random.uniform(0, 1)
            if mistake_indicator < u:
                #print("There has been a mutation!")
                mutation_list = [x for x in strat_list if x != reproduced_strategy]
                if mutation_list == []:
                    #print("There cannot be mutations in this population.")
                    G.node[j]['strategy'] = reproduced_strategy
                else:
                    G.node[j]['strategy'] = np.random.choice(mutation_list)
                    reproduced_strategy = G.node[j]['strategy']
                #print("Node ", j, " now has strategy ", G.node[j]['strategy'])        
            else:
                # there is not a mutation
                G.node[j]['strategy'] = reproduced_strategy
            # Node j has now just been born, so we set its fitness to 0
            G.node[j]['fitness'] = 0.5
            reproduced_strategies.append(reproduced_strategy)
            old_strategies.append(old_strategy)
    return [G, reproduced_strategies, old_strategies, reproduced_nodes]

    '''
    # Iterate through all nodes of the graph
    # Note, i is a coordinate whose dimension depends on the dimension of the graph
    current_place_in_sum = 0
    prev_place_in_sum = 0
    node_reproduced = False 
    for i in nx.nodes(G):
        current_place_in_sum += nx.get_node_attributes(G, 'fitness')[i]
        #G.node[i]['fitness']
        #print("Currently examining node ", i, " which has fitness ", G.node[i]['fitness'])
        #print("Current place in the fitness sum is ", current_place_in_sum)
        if prev_place_in_sum < cutoff < current_place_in_sum:
            node_reproduced = True
            #print("We have found a node to reproduce")
            #print("        It is node ", i, " has fitness ", G.node[i]['fitness'], " and strategy ", G.node[i]['strategy'])
            #This is the node that will reproduce
            #so we now find a neighbor it can replace
            reproduced_strategy = G.node[i]['strategy'] 
            #print("Node ", i, " has been chosen to reproduce strategy ", reproduced_strategy)

            #print("-----------------------------------")
            neighbors = list(G.adj[i].keys())
            #print("Neighbors are ", neighbors)
            num_neighbors = len(neighbors)
            k = random.randint(0, num_neighbors - 1)
            j = neighbors[k]
            old_strategy = G.node[j]['strategy']
            # set the strategy of the node selected to die to the strategy of the node selected to reproduce
            #print("Updating the strategy of node ", j, " from ", old_strategy, " to ", reproduced_strategy)
                

            mistake_indicator = random.uniform(0, 1)
            if mistake_indicator < u:
                # there is a mutation
                #print("There has been a mutation!")
                mutation_list = [x for x in strat_list if x != reproduced_strategy]
                if mutation_list == []:
                    #print("There cannot be mutations in this population.")
                    G.node[j]['strategy'] = reproduced_strategy
                else:
                    G.node[j]['strategy'] = np.random.choice(mutation_list)
                #print("Node ", j, " now has strategy ", G.node[j]['strategy'])
                     
            else:
                # there is not a mutation
                G.node[j]['strategy'] = reproduced_strategy

            # Node j has now just been born, so we set its fitness to 0
            G.node[j]['fitness'] = 0.5
            #print("------------------------------------------------------------------------")
            #print("reproduction has finished for this round")
            # no need to examine any more nodes for reproducibility, so we break our for loop
            break
        prev_place_in_sum = current_place_in_sum

    # Creates picture of graph 
    #nx.draw(G,with_labels=True)
    #plt.show()
    if node_reproduced:
        return [G, G.node[j]['strategy'], old_strategy]
    else:
        return [G, None, None]
    '''

def death_birth(G, strat_list, u):
    replaced=random.choice(list(G.nodes()))
    old_strategy=nx.get_node_attributes(G, 'strategy')[replaced]    #G.node[replaced]['strategy']
    #competition dictionary through which neighbors will compete
    weights=nx.get_edge_attributes(G, 'weight')
    fits=nx.get_node_attributes(G, 'fitness')
    competition={}

    for n in list(G.neighbors(replaced)):
        possible_orders=[(n,replaced), (replaced,n)]
        for order in possible_orders:
            try:
                competition[n]=fits[n]* weights[order]
            except:
                pass

        #m=min([n,replaced])
        #M=max([n,replaced])
        #competition[n]=fits[n]* weights[(m,M)]
        #G.node(n)['fitness']* weights[(n,replaced)]
    sum_competition=sum(competition.values())
    cutoff=random.uniform(0,sum_competition)
    sum_so_far=0
    for node in G.neighbors(replaced):
        sum_so_far+=competition[node]
        if cutoff<sum_so_far:
            reproduced=node
    #now place reproduced into the replaced node, 
    #consider mutation
    mistake_indicator = random.uniform(0, 1)
    if mistake_indicator<u:
        #there is a mutation
        G.node[replaced]['strategy']=random.choice(strat_list)
    else:
        G.node[replaced]['strategy']=G.node[reproduced]['strategy']
    return [G, G.node[replaced]['strategy'], old_strategy]

def pairwise_comparison(G, strat_list, u):
    #Like above, choose uniformly for replaced
    replaced=random.choice(nx.nodes(G))
    old_strategy=G.node[replaced]['strategy']
    #competition with e_ij between neighbors
    weights=nx.get_edge_attributes(G, 'weight')
    competition={}
    for n in G.neighbors(replaced):
        competition[n]=weights[(n,replaced)]
    sum_competition=sum(competition.values())
    cutoff=random.uniform(0,sum_competition)
    sum_so_far=0
    for node in G.neighbors(replaced):
        sum_so_far+=competition[node]
        if cutoff<sum_so_far:
            reproduced=node
    #replacement with probability Theta(F_i-F_j)
    difference=G.node[reproduced]['fitness']-G.node[replaced]['fitness']
    if random.random()<=q(difference, .1, 3):
    #if random.random()<=Theta(difference):
        #place reproduced into replaced,consider mutation
        mistake_indicator = random.uniform(0, 1)
        if mistake_indicator<u:
            #there is a mutation
            G.node[replaced]['strategy']=random.choice(strat_list)
        else:
            G.node[replaced]['strategy']=G.node[reproduced]['strategy']
    return [G, G.node[replaced]['strategy'], old_strategy]

def imitation(G, strat_list, u):
    #Choose node uniformly to be replaced
    replaced=random.choice(nx.nodes(G))
    old_strategy=G.node[replaced]['strategy']
    #competition with e_ij between neighbors
    weights=nx.get_edge_attributes(G, 'weight')
    competition={}
    for n in G.neighbors(replaced):
        competition[n]=G.node[n]['fitness']
    sum_competition=sum(competition.values())
    cutoff=random.uniform(0,sum_competition)
    sum_so_far=0
    for node in G.neighbors(replaced):
        sum_so_far+=competition[node]
        if cutoff<sum_so_far:
            reproduced=node
    #place reproduced into replaced,consider mutation
    mistake_indicator = random.uniform(0, 1)
    if mistake_indicator<u:
        #there is a mutation
        G.node[replaced]['strategy']=random.choice(strat_list)
    else:
        G.node[replaced]['strategy']=G.node[reproduced]['strategy']
    return [G, G.node[replaced]['strategy'], old_strategy]

# The following code may be modified to ''
'''
def color_graph(G):
    #INPUTS:     A graph object

    #OUTPUTS:    The same graph but with color attributes 
    #            according to strategies at each node

    # create empty list for node colors
    node_color = []

    # iterate through each node in the graph
    for node in G.nodes(data=True):

        # iterate through all strategyies
        for strat in nx.get_node_attributes(G, 'strategy'):

            # if the node has the attribute group1
            if 'cooperate' in node[1]['strategy']:
                node_color.append('blue')

            # if the node has the attribute group1
            elif 'group2' in node[1]['group']:
                node_color.append('red')

            # if the node has the attribute group1
            elif 'group3' in node[1]['group']:
                node_color.append('green')

            # if the node has the attribute group1
            elif 'group4' in node[1]['group']:
                node_color.append('yellow')

            # if the node has the attribute group1
            elif 'group5' in node[1]['group']:
                node_color.append('orange')  

    # draw graph with node attribute color
    nx.draw(g, with_labels=False, node_size=25, node_color=node_color)
    return G
'''


def Theta(val):
    V=1+math.exp(-val)
    return 1/V

def q(val, K, d):
    V = 1+ math.exp(val/(K*d))
    return 1/V 

