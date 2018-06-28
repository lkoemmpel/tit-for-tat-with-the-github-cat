import networkx as nx
import random
import matplotlib.pyplot as plt

'''
TESTING GRAPH 1
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

b=2
c=1
this_lambda=0.5
kappa=0.5
omega0=0.5

def f(val):
     return (1+exp(-kappa*(val-val0)))**(-1)

'''
INTERACTION CLASS
'''

class interaction():
     '''
     To realize all the pertinent pairwise interactions in a turn 
     '''
     def __init__(self, G):
          self.G = G

     def general_reciprocity_simple(G, set_nodes, b, c, f, asynchronous=False): 
          #realize all of the interactions between pairs of players belonging to the set
          for v in set_nodes:
               requested=random.choice(G.adj[v].keys())
               #interact1(v,w)
               p=random.random()
               #If the node w decides to help (with probability p_w(t)
               if p<=G.node[requested]['coop-state']:
                    #update payoffs, fitness, probabilities, etc for next timestep
                    G.node[requested]['turn-payoff']-=c
                    G.node[v]['turn-payoff']+=b
                    G.node[v]['total-payoff']+=G.node[v]['turn-payoff']
               if asynchronous:
                    A=(G.node[v]['coop-state'])**(1-this_lambda)
                    B=(f(G.node[v]['total-payoff']))**(this_lambda)
                    G.node[v]['coop-state']=A*B
               else:
                    G.node[v]['coop-state']=f(G.node[v]['total-payoff'])
               G.node[v]['turn-payoff']=0
          #return modified graph 
          return G  

     def indirect_reciprocity(G, set_nodes, b, c, f):
          #realize all of the interactions between pairs of players belonging to the set
          for v in set_nodes:
               requested=random.choice(G.adj[v].keys())

               G.node[v]['turn-payoff']=#b*x_requested-c*x_v*sum_(nbhd of i)[rho_k at t]


               #interact1(v,w)
               p=random.random()
               #If the node w decides to help (with probability p_w(t)
               if p<=G.node[requested]['coop-state']:
                    #update payoffs, fitness, probabilities, etc for next timestep
                    G.node[requested]['turn-payoff']-=c
                    G.node[v]['turn-payoff']+=b
                    G.node[v]['total-payoff']+=G.node[v]['turn-payoff']
               if asynchronous:
                    A=(G.node[v]['coop-state'])**(1-this_lambda)
                    B=(f(G.node[v]['total-payoff']))**(this_lambda)
                    G.node[v]['coop-state']=A*B
               else:
                    G.node[v]['coop-state']=f(G.node[v]['total-payoff'])
               G.node[v]['turn-payoff']=0
          #return modified graph 
          return G  

