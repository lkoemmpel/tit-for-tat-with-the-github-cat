import igraph
import random

def timestep_control(G):
  #determines when nodes should be deleted or reproduce according to probability distribution (e.g. poisson)
