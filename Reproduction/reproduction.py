import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sys


'''-------------------
        IMPORTS
-------------------'''
import sys
sys.path.insert(0, r'/Users/olgachuy/Downloads/SPUR+/tit-for-tat-with-the-github-cat/graph_initialization')
import graph_initialization as init



# add inherited fitness
# create probabilistic cooperative state from Utkovsky paper
# add mutation affecting reproduction
# replicate early results like pop density paper
# do work on 3d lattice






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



class strategy_update():

    '''
    make all of the reproduction process based on the fitness, weight and mutation parameters.

    '''

    # iterate through all nodes of the graph
    # choose a random node proportinal to its fitness to reproduce
    def __init__(self, G, u, strat_list):
        self.G = G
        self.u = u
        self.strat_list = strat_list

    def birth_death(self):
        fitness_dict = nx.get_node_attributes(G, 'fitness')
        #print("The fitness dictionary is ", fitness_dict)
        fitness_sum = sum(fitness_dict.values())
        #print("The fitness sum is ", fitness_sum)

        cutoff = random.uniform(0, fitness_sum)
        #print("The cutoff is ", cutoff)

        # Iterate through all nodes of the graoh
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
                if mistake_indicator < self.u:
                    # there is a mutation
                    print("There has been a mutation!")
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
        return [self.G, G.node[j]['strategy'], old_strategy]

    def death_birth(self):
        return self.G

    def pairwise_comparison(self):
        return self.G

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

def plot_proportion_data(time, strat_dict):
    for strat in strat_dict:
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
        plt.xlabel('Proportion of nodes with strategy ' + strat)
        plt.ylabel('Time')

        #show plot
        plt.show()

    return None


def run_simulation(G, u, t, plotting = False, show_graph = False):
    '''
    INPUTS:     G: networkx graph object with fitness and strategy attributes
                u: rate of mutation for reproduction
                t: number of times to have stratgies update

    OUTPUTS:    new_graph: updated networkx graph object where strategies have been updated

    Prints graph at every stage of strategy updating
    Plots how proportions of each strategy change over time
    '''
    if show_graph:
        color_and_draw_graph(G)
        print(nx.get_node_attributes(G, 'strategy'))
        print("-----------------------------------------------")

    if plotting:
        time_data = [0]
        final_data = {}
        proportions_dict = {}
        strat_data_dict = {}
        # initialize strategy tallies to 0
        for strat in strat_list:
            strat_data_dict[strat] = 0
            final_data[strat] = []

        # if a strategy is found in the graph, adjust its tally
        for n in nx.nodes(G):
            strat_data_dict[G.node[n]['strategy']] += 1
        for strat in strat_data_dict:
            final_data[strat].append(strat_data_dict[strat]/nx.number_of_nodes(G))


    


    for i in range(t):
        graph = strategy_update(G, u, strat_list)

        birth_death_results = graph.birth_death()
        new_graph = birth_death_results[0]
        new_strategy = birth_death_results[1]
        old_strategy = birth_death_results[2]

        #print(nx.get_node_attributes(G, 'strategy'))
        #print('\n')

        if show_graph:
            # Creates picture of graph 
            color_and_draw_graph(new_graph)


        if plotting:
            # update tallies for each strategy
            strat_data_dict[new_strategy] += 1
            strat_data_dict[old_strategy] -= 1

            # update strategy proportions for each strategy
            for strat in strat_data_dict:
                final_data[strat].append(strat_data_dict[strat]/nx.number_of_nodes(G))

            #print("Current time data is", time_data)
            time_data.append(i+1)

        #print(time_data)
        #print(final_data)
    plot_proportion_data(time_data, final_data)

    #print(new_graph.adj)
    return new_graph

run_simulation(G, .2, 30, plotting = True, show_graph = False)