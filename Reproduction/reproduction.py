import networkx as nx
import random
import matplotlib.pyplot as plt




# add inherited fitness
# create probabilistic cooperative state from Utkovsky paper
# add mutation affecting reproduction
# replicate early results like pop density paper
# do work on 3d lattice




import sys
#sys.path.append()


color_map = []

# Initialization of graph to use for testing purposes
G = nx.triangular_lattice_graph(4,3)
for n in nx.nodes(G):
    # Assign node attributes
    # Initial strategy is C or D, chosen randomly
    strategy_indicator = random.randrange(0, 2, 1)
    if strategy_indicator == 1:
        G.node[n]['strategy'] = 'Cooperate'
        color_map.append('blue')
    else:
        G.node[n]['strategy'] = 'Defect'
        color_map.append('red')
    # fitness attributes are chosen randomly
    # print("The current strategy of node ", n, " is ", G.node[n]['strategy'] )
    G.node[n]['fitness'] = random.uniform(0,1)


strat_list = ['Cooperate', 'Defect']



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

                #print("-----------------------------------")
                neighbors = list(G.adj[i].keys())
                #print("Neighbors are ", neighbors)
                num_neighbors = len(neighbors)
                k = random.randint(0, num_neighbors - 1)
                j = neighbors[k]
                # set the strategy of the node selected to die to the strategy of the node selected to reproduce
                print("Updating the strategy of node ", j, " from ", G.node[j]['strategy'], " to ", reproduced_strategy)
                

                mistake_indicator = random.uniform(0, 1)
                if mistake_indicator < u:
                    # there is a mutation
                    print("There has been a mutation!")
                    mutation_list = [x for x in strat_list if x != reproduced_strategy]
                    if mutation_list == []:
                        print("There cannot be mutations in this population.")
                    else:
                        G.node[n]['strategy'] = numpy.random.choice(mutation_list)
                     
                else:
                    # there is not a mutation
                    G.node[j]['strategy'] = reproduced_strategy
                print("------------------------------------------------------------------------")
                #print("reproduction has finished for this round")
                # no need to examine any more nodes for reproducibility, so we break our for loop
                break
            prev_place_in_sum = current_place_in_sum

        # Creates picture of graph 
        #nx.draw(G,with_labels=True)
        #plt.show()
        return self.G

    def death_birth(self):
        return self.G

    def pairwise_comparison(self):
        return self.G


def color_graph(G):
    '''
    INPUTS:     A graph object

    OUTPUTS:    The same graph but with color attributes 
                according to strategies at each node
    '''

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



t = 5
u = .2
for i in range(t):
    graph = strategy_update(G, u, strat_list)
    new_graph = graph.birth_death()
    # Creates picture of graph 
    print(nx.get_node_attributes(G, 'strategy'))
    nx.draw(G,node_color = color_map,with_labels = True)
    plt.show()

#print(new_graph.adj)
    