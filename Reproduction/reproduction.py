import networkx as nx
import random

G = nx.triangular_lattice_graph(10,20)
for n in self.G.nodes_iter:
	fitness = random.uniform(0,1)
	current_strat = random.randrange(0, 1, 1)





	# associate to each node of the graph a tuple of its fitness and current state (C or D)
	set_node_attributes(G, (fitness, strategy), n)

print(G.adj)
nx.draw(G,with_labels=True)
plt.show()


class strategy_update():

	'''
	make all of the reproduction process based on the fitness, weight and mutation parameters.

 	'''

	# iterate through all nodes of the graph
	# with probability p, consider allowing a node to reproduce
	def __init__(self, G):
		self.G = G


	def birth_death(self):
        fitness_dict = get_node_attributes(G, fitness)
        fitness_sum = sum(fitness_dict.values())

        cutoff = randrange(0, fitness_sum)

        for i in G.number_of_nodes():
            if cutoff < fitness_dict[i]:
                #This node will reproduce
                #so we now find a neighbor it can replace
                reproduced_strategy = get_node_attributes(G, strategy, i)

                num_neighbors = len(all_neighbors(G,i))
                death_index = random.uniform(0,num_neighbors)
                # set the strategy of the node selected to die to the strategy of the node selected to reproduce
                set_node_attributes(G, reproduced_strategy, all_neighbors(G, i)[death_index])


	    nx.draw(G,with_labels=True)
		plt.show()
    	




        return self.name

    def death-birth(self):
        return self.species

    def pairwise-comparison(self):
    	return G

    
  	