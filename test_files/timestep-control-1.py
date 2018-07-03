
'''-------------------
        PACKAGES
-------------------'''
import random
import networkx as nx
import matplotlib.pyplot as plt
import math 
import numpy as np 

'''-------------------
        IMPORTS
-------------------'''
import graph_initialization as init
import reproduction as rep
import interaction as inter

'''
class gameFIRST_TRY():
    def __init__(self, strat_update, name):
        #initialize the game with the parameters: 
        #graph
        #time
        #name of game played 
        self.graph=G
        self.name=name
        #self.t=0

    #When is this used/what is this for??
    
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
    
'''


#strat_list = ['Cooperate', 'Defect', 'Tit_for_tat']
strat_list = ['Cooperate', 'Defect']
b = 2
c = 1
delta = 0


class game():
    def __init__(self, graph, name, t, u=0, delta=0, plotting = False, show_graph = False ):
        self.graph=G
        #name of the game
        self.name=name

        #What does this do????????
        #self.strategy_update=strategy_update(graph, u, strat_list)
        #self.fitness_update=interaction_process(graph, 2, 1)

        self.u = u
        self.delta = delta
        self.t = t
        self.plotting = plotting
        self.show_graph = show_graph

    def timestep(G, u, t, name = 'BD', plotting = False, show_graph = False):
        '''
        INPUTS:     G: networkx graph object with fitness and strategy attributes
                    u: rate of mutation for reproduction
                    t: number of times to have stratgies update

        OUTPUTS:    new_graph: updated networkx graph object where strategies have been updated

        Prints graph at every stage of strategy updating
        Plots how proportions of each strategy change over time
        '''
        if name=='BD':
            if show_graph:
                rep.color_and_draw_graph(G)
                #print(nx.get_node_attributes(G, 'strategy'))
                #print("-----------------------------------------------")

            if plotting:
                time_data = [0]
                final_data = {}
                proportions_dict = {}
                strat_data_dict = {}
                # initialize strategy tallies to 0
                for strat in strat_list:
                    strat_data_dict[strat] = 0
                    final_data[strat] = []

                # if a strategy is found in the graph, adjust its tally
                for n in nx.nodes(G):
                    strat_data_dict[G.node[n]['strategy']] += 1
                for strat in strat_data_dict:
                    final_data[strat].append(strat_data_dict[strat]/nx.number_of_nodes(G))


            for i in range(t):
                #adjust node strategies 
                graph = rep.strategy_update(G, u, strat_list)
                # TODO-----------------------------
                # figure out why variables have to be directly input
                # into the birth-death function
                birth_death_results = graph.birth_death(G, strat_list)
                new_graph = birth_death_results[0]
                new_strategy = birth_death_results[1]
                old_strategy = birth_death_results[2]

                #play the game
                payoff_mtx = [ [(b-c, b-c), (-c, b)] , [(b, -c), (0,0)] ]
                coop_index={'Cooperate':0, 'Defect':1}
                new_graph = inter.interaction_process(new_graph, b, c).interaction_BD(payoff_mtx, delta, noise=0)

                #print(nx.get_node_attributes(G, 'strategy'))
                #print('\n')

                if show_graph:
                    # Creates picture of graph 
                    rep.color_and_draw_graph(new_graph)


                if plotting:
                    # update tallies for each strategy
                    strat_data_dict[new_strategy] += 1
                    strat_data_dict[old_strategy] -= 1

                    # update strategy proportions for each strategy
                    for strat in strat_data_dict:
                        final_data[strat].append(strat_data_dict[strat]/nx.number_of_nodes(G))

                    #print("Current time data is", time_data)
                    time_data.append(i+1)

                    #print(time_data)
                    #print(final_data)
                    print("Plotting data at time ", t)
                    rep.plot_proportion_data(time_data, final_data)

                #print(new_graph.adj)
                graph = new_graph
            return new_graph


'''--------------------------
        SIMULATION 1
--------------------------'''
G=init.generate_lattice(10,10)
init.label_strategies(G, strat_list)
init.label_fitness(G)
game.timestep(G, u=.2, t=10, name= 'BD', plotting = True, show_graph = True)


'''
#Initialize a graph
G=generate_lattice(4,5)
game1=game(G, 'DB', 0.5, 0)
#run birth death process 1000 times
for t in range(1000):
    game1.timestep()
'''

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
#   SF+=F[i]
#   S.append(F[i])
# C=[1 for i in range(n)]


# for t in range(T):
#   G=timestep_control(G)
# nx.draw(hg)
# plt.show()





  #D={'C':color1, 'D':color2, 'T':color3}
  #for v in the graph:

