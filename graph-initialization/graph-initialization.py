'''
INPUTS: 
  Number of nodes
  Number of cliques
  Type of graph

OUTPUTS:
  Some form of a graph
'''


import igraph
pip install networkx
n=10000
m=10000
G1=networkx.barbarasi_albert_graph(n,m[,seed])
d=70
G2=networkx.random_egular_graph(n,d[,seed])
