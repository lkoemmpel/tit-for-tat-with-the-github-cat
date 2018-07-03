'''-------------------
        PACKAGES
-------------------'''

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sys


'''-------------------
        IMPORTS
-------------------'''
'''
import sys
#sys.path.insert(0, r'/Users/olgachuy/Downloads/SPUR+/tit-for-tat-with-the-github-cat/graph_initialization')
sys.path.insert(0, r'C:/Users/Laura/Documents/MIT/Summer stuff 2018/Github work - SPUR 2018/tit-for-tat-with-the-github-cat/graph-initialization')
'''
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

def birth_death(G, strat_list, u):
    # choose a random node proportinal to its fitness to reproduce

    fitness_dict = nx.get_node_attributes(G, 'fitness')
    #print("The fitness dictionary is ", fitness_dict)
    fitness_sum = sum(fitness_dict.values())
    #print("The fitness sum is ", fitness_sum)
    cutoff = random.uniform(0, fitness_sum)
    #print("The cutoff is ", cutoff)

    # Iterate through all nodes of the graph
    # Note, i is a coordinate whose dimension depends on the dimension of the graph
    current_place_in_sum = 0
    prev_place_in_sum = 0
    for i in nx.nodes(G):
        current_place_in_sum += G.node[i]['fitness']
        #print("Currently examining node ", i, " which has fitness ", G.node[i]['fitness'])
        #print("Current place in the fitness sum is ", current_place_in_sum)
        if prev_place_in_sum < cutoff < current_place_in_sum:
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
            G.node[j]['fitness'] = 0
            #print("------------------------------------------------------------------------")
            #print("reproduction has finished for this round")
            # no need to examine any more nodes for reproducibility, so we break our for loop
            break
        prev_place_in_sum = current_place_in_sum

    # Creates picture of graph 
    #nx.draw(G,with_labels=True)
    #plt.show()
    return [G, G.node[j]['strategy'], old_strategy]

def death_birth(G):
    return 

def pairwise_comparison(G):
    return 

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

def color_and_draw_graph(G):
    '''
    INPUTS:     Graph with strategy node attributes
    OUTPUTS:    Graph where color maps has colors according to 
                node strategy attributes
    '''
    # initializes color map
    color_map = []
    for n in nx.nodes(G):
        if G.node[n]['strategy'] == 'Cooperate':
            color_map.append('green')
        else:
            color_map.append('red')

    # draws colored graph
    #plt.ion()
    nx.draw(G,node_color = color_map,with_labels = True)
    plt.show()
    #plt.pause(2.0)

    return G

def plot_proportion_data(time, strat_dict, save=False):
    for strat in strat_dict:

        # IMPORTANT -- UNDO THIS IF STATEMENT WHEN EXAMINING MORE STRATEGIES
        if strat == 'Cooperate':
            #scatter plot
            X = time
            Y = strat_dict[strat]
            plt.plot(X, Y, color='blue', marker='^', linestyle = '-')

            #change axes ranges
            plt.xlim(0,max(time))
            plt.ylim(0,1)

            #add title
            plt.title('Relationship between time and proportion of nodes with strategy ' + strat)

            #add x and y labels
            plt.ylabel('Proportion of nodes with strategy ' + strat)
            plt.xlabel('Time')

            #show plot
            plt.show()

    return None


