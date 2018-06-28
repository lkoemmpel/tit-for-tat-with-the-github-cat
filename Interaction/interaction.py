import igraph
import random
import networkx as nx

b=2
c=1
kappa=0.5
val0=0.5
def f(val):
  return (1+exp(-kappa*(val-val0)))**(-1)

def interaction1(G, set_nodes, b, c, f, ):
  for v in set_nodes:
    N=G[v]
    attr=nx.get_node_attributes(G)



def interaction2(G, set_nodes, b, c, f, asynchronous=False): 
  #realize all of the interactions between pairs of players belonging to the set
  for v in set_nodes:
    N=G[v]
    w=random.choice(N)
    #interact1(v,w)
    p=random.random()
    #If the node w decides to help (with probability p_w(t)
    if p<=w.prob(t):
      #update payoffs, fitness, probabilities, etc for next timestep
      w.y+=b-c
      v.y+=b
      v.Y+=v.y(t)
      if asynchronous:
<<<<<<< HEAD
=======

      else:  
        v.prob=f(v.Y)
  #return modified graph 
  return G 
>>>>>>> 059d7ea20a79512ff69a385dd60f593993de8916

      else:  
        v.prob=f(v.Y)
  #return modified graph 
  return G 