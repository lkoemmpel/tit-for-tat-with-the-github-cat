
import random
import networkx as nx

b=2
c=1
kappa=0.5
val0=0.5
def f(val):
  return (1+exp(-kappa*(val-val0)))**(-1)

def interaction(G, set_nodes, b, c, f): 
  #realize all of the interactions between pairs of players belonging to the set
  for v in set_nodes:
    N=G[v]
    w=random.choice(N)
    #interact1(v,w)
    p=random.random()
    #If the node w decides to help (with probability p_w(t)
    if p<=w.prob(t):
      #update payoffs, fitness, probabilities, etc for next timestep
      w.y(t)+=b-c
      v.y(t+1)=b
      v.Y(t)=v.Y(t-1)+v.y(t)
      v.prob(t+1)=f(v.Y(t))
  #return modified graph 
  return G 
  

