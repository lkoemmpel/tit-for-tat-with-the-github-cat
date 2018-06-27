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

def generate_lattice(n, m, type, dim, periodic = False, with_positions = True, create_using = None):
	if type == 'triangular':
		lattice = nx.triangular_lattice_graph(10, 20, periodic=False, create_using=None)
		nx.draw(lattice)
		plt.show()
	return lattice



def generate_graph(n, m, type = hypercube, periodic=False, with_positions=True, create_using=None):
	
	if type == 'hypercube':
		graph = nx.hypercube_graph(i)
	else:
		raise NameError

	nx.draw(hg)

	plt.show()

	return graph


