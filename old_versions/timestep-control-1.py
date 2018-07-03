
'''-------------------
		PACKAGES
-------------------'''

import igraph
import random
import networkx as nx
import matplotlib.pyplot as plt
import importlib

'''-------------------
		IMPORTS
-------------------'''

import graph_initialization as init
import reproduction as rep
import interaction as inter

class gameFIRST_TRY():
	def __init__(self, strat_update, name):
		#initialize the game with the parameters: 
		#graph
		#time
		#name of game played 
		self.graph=G
		self.name=name
		self.t=0
	def step_BD(self):
		#standard kind of step
		G=self.graph
		g=strategy_update(G,u,strat_list)

		G=interaction(G)
		G=reproduction(G)
		return G
	def set_nodes(self, game='BD'):
		G=self.graph
		if game=='BD':
			return set(G.adj.keys())

	def choose_node_weighted_F(G, W):
		nodes=G.nodes
		x=random.choice(nodes, [W[i]/SF for i in range(n)])
		return x

import igraph
import random
import networkx as nx
import matplotlib.pyplot as plt
import importlib

list_strategies=['Cooperate', 'Defect', 'Tit_for_tat']


class game():
	def __init__(self, graph, name, u=None, delta=None, ):
		self.graph=graph
		#name of the game
		self.name=name
		self.strategy_update=strategy_update(graph, u, strat_list)
		self.fitness_update=interaction_process(graph, 2, 1)
		self.time=0
	def timestep(self):
		if name=='BD':
			self.time+=1
			G=self.strategy_update.birth_death()
			G=self.interaction_process.birth_death()
			self.graph=G

'''--------------------------
		SIMULATION 1
--------------------------'''

#Initialize a graph
G=generate_lattice(4,5)
game1=game(G, 'DB', 0.5, 0)
#run birth death process 1000 times
for t in range(1000):
	game1.timestep()


# n=20
# m=30
# T=100000
# G=generate_lattice(n, m, 'triangular', 4)
# plt.show()

'''
INITIALIZE GRAPHS
'''

# #get the fitness values of the vertices in the graph
# SF=0
# S=[0]
# for v in G:

# F=[random.randrange(1,2) for i in range(n)]
# for i in range(n):
# 	SF+=F[i]
# 	S.append(F[i])
# C=[1 for i in range(n)]


# for t in range(T):
# 	G=timestep_control(G)
# nx.draw(hg)
# plt.show()





  #D={'C':color1, 'D':color2, 'T':color3}
  #for v in the graph:

