import igraph
import random
import networkx as nx
import matplotlib.pyplot as plt
import importlib

importlib.__import__('graph-initialization')
importlib.__import__('interaction')
importlib.__import__('reproduction')


def choose_node_weighted_F(G, W):
	nodes=G.nodes
	x=random.choice(nodes, [W[i]/SF for i in range(n)])
	return x
	# for i in range(n):
	# 	if W[i]<x<W[i+1]:
	# 		chosen=i+1

def timestep_control(G):
	#determines when nodes should be deleted or reproduce according to probability distribution (e.g. poisson)
	i=choose_node_weighted_F(G, S) #random vertex in G, use F_i
	j=choose_node_weighted_F(G, C) #uniform random choice of vertex in G
	#interaction and reproduction
	G=interaction(G, i, j)
	G=reproduction(G)
	return G

n=20
m=30
T=100000

G=generate_lattice(n, m, 'triangular', 4)
plt.show()

#INITIALIZE GRAPH

#get the fitness values of the vertices in the graph
SF=0
S=[0]
for v in G:

F=[random.randrange(1,2) for i in range(n)]
for i in range(n):
	SF+=F[i]
	S.append(F[i])
C=[1 for i in range(n)]


for t in range(T):
	G=timestep_control(G)
nx.draw(hg)
plt.show()

# class Node(object):
#   def __init__(self, prob, y, Y):
#     self.prob=prob
#     self.y=0
#     self.Y=0