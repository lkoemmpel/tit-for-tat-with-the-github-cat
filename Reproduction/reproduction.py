import networkx as nx
import random
import matplotlib.pyplot as plt

# Initialization of graph to use for testing purposes
G = nx.triangular_lattice_graph(4,3)
for n in nx.nodes(G):
    # Assign node attributes
    # Initial strategy is C or D, chosen randomly
    strategy_indicator = random.randrange(0, 1, 1)
    if strategy_indicator == 1:
        G.node[n]['strategy'] = 'Cooperate'
    else:
        G.node[n]['strategy'] = 'Defect'
    # fitness attributes are chosen randomly
    # print("The current strategy of node ", n, " is ", G.node[n]['strategy'] )
    G.node[n]['fitness'] = random.uniform(0,1)



class strategy_update():

    '''
    make all of the reproduction process based on the fitness, weight and mutation parameters.

    '''

    # iterate through all nodes of the graph
    # choose a random node proportinal to its fitness to reproduce
    def __init__(self, G):
        self.G = G
    def birth_death(self):
        fitness_dict = nx.get_node_attributes(G, 'fitness')
        print("The fitness dictionary is ", fitness_dict)
        fitness_sum = sum(fitness_dict.values())
        print("The fitness sum is ", fitness_sum)

        cutoff = random.uniform(0, fitness_sum)
        print("The cutoff is ", cutoff)

        # Iterate through all nodes of the graoh
        # Note, i is a coordinate whose dimension depends on the dimension of the graph
        for i in nx.nodes(G):
            if G.node[i]['fitness'] < cutoff < G.node[i]['fitness']:
                #This is the node that will reproduce
                #so we now find a neighbor it can replace
                reproduced_strategy = G.node[i]['strategy'] 

                num_neighbors = len(all_neighbors(G,i))
                j = randint(0, num_neighbors)
                # set the strategy of the node selected to die to the strategy of the node selected to reproduce
                print("Updating the strategy of node ", j, " from ", G.node[j]['strategy'], " to ", reproduced_strategy)
                G.node[j]['strategy'] = reproduced_strategy
                print("reproduction has finished for this round")

        # Creates picture of graph 
        nx.draw(G,with_labels=True)
        plt.show()
        return self.G

    def death_birth(self):
        return self.G

    def pairwise_comparison(self):
        return self.G

graph = strategy_update(G)
new_graph = graph.birth_death()
print(new_graph.adj)
    