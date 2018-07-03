'''
INPUTS: 
  Number of nodes
  Number of cliques
  Type of graph

OUTPUTS:
  Some form of a graph
'''


import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_lattice(n, m, type = 'triangular', dim = 2, periodic = False, with_positions = True, create_using = None):
    try:
        if type == 'triangular':
            lattice = nx.triangular_lattice_graph(10, 20, periodic=False, create_using=None)
            #nx.draw(lattice)
            #plt.show()
            return lattice
    except ValueError:
        print("The specified lattice type was invalid.")



def generate_graph(n, m, type = 'hypercube', periodic=False, with_positions=True, create_using=None):
	try:
		if type == 'hypercube':
			graph = nx.hypercube_graph(i)
		else:
			raise NameError

		nx.draw(hg)

		plt.show()

		return graph
	except ValueError:
		print("The specified graph type was invalid.")

print(generate_lattice(3,4))


def label_strategies(G, strat_list):
  for n in nx.nodes(G):
    G.node[n]['strategy']=random.choice(strat_list)

def label_fitness(G):
  for n in nx.nodes(G):
    G.node[n]['fitness'] = random.uniform(0,1)

