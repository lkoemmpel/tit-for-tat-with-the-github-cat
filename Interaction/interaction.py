import igraph
import random

b=2
c=1
kappa=0.5
val0=0.5
def f(val):
  return (1+exp(-kappa*(val-val0)))**(-1)

def interaction(G, set_nodes, b, c, f): 
  #realize all of the interactions between pairs of players belonging to the set
  for v in set_nodes:
    N=neighborhood(v)
    w=random.choice(N)
    interact1(v,w)
    p=random.random()
    if p<=w.prob(t):
      w.y(t)+=b-c
      v.y(t)+=b
      v.Y(t)=v.Y(t-1)+v.y(t)
      v.prob(t+1)=f(v.Y(t))
  #return modified graph 
  return G 
  

