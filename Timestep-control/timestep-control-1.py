import igraph
import random
import networkx as nx
import matplotlib.pyplot as plt
import importlib

importlib.__import__('graph-initialization')
importlib.__import__('interaction')
importlib.__import__('reproduction')

class game():
	def __init__(self, G, game):
		self.graph=G
		self.game=game
		self.t=0
	def step(self):
		G=self.graph
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

n=20
m=30
T=100000
G=generate_lattice(n, m, 'triangular', 4)
plt.show()

'''
INITIALIZE GRAPHS
'''

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

'''
GRAPH 1
'''
G=nx.triangular_lattice_graph(5,6)
for v in nx.nodes(G):
  #assign strategy with probability 1/2
  choice=random.choice([0,1])
  if choice==0:
    G.node[v]['strategy']='Cooperate'
  else:
    G.node[v]['strategy']='Defect'
  #itness takes some random value from 0 to 1
  G.node[v]['fitness']=random.uniform(0,1)
  #cooperative_state = probability of helping another node
  G.node[v]['coop-state']=random.uniform(0,1)

  #payoff from interactions
  G.node[v]['turn-payoff']=0
  G.node[v]['total-payoff']=0
