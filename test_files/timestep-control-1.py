
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


class game():
    def __init__(self, graph, name, t, u=0, delta=0, plotting = False, show_graph = False ):
        self.graph=G
        #name of the game
        self.name=name

        #What does this do?
        #self.strategy_update=strategy_update(graph, u, strat_list)
        #self.fitness_update=interaction_process(graph, 2, 1)

        self.u = u
        self.delta = delta
        self.t = t
        self.plotting = plotting
        self.show_graph = show_graph

    def trial(self, G, n, d, data_iteration, u, t, graph_type = 'random', update_name = 'BD', plotting = False, show_graph = False, saving = False):
        '''
        INPUTS:     G: networkx graph object with fitness and strategy attributes
                    u: rate of mutation for reproduction
                    t: number of times to have stratgies update

        OUTPUTS:    new_graph: updated networkx graph object where strategies have been updated

        Prints graph at every stage of strategy updating
        Plots how proportions of each strategy change over time
        '''
        if update_name=='BD':

            if show_graph:
                rep.color_and_draw_graph(G)
                #print(nx.get_node_attributes(G, 'strategy'))
                #print("-----------------------------------------------")

            if plotting:
                time_data=[]

                strat_data_dict, concentrations= get_histogram_and_concentration_dict(G, strat_list)
                # strat_data_dict maps      strategy ---> freq(strat)
                # concentrations maps       strategy ---> matrix 
                #                                         entry n,t : freq(strat)/number(nodes)
                #                                         at trial n, time t

            for i in range(t):

                #adjust node strategies 
                birth_death_results = rep.birth_death(G, strat_list, u)
                new_graph = birth_death_results[0]
                new_strategy = birth_death_results[1]
                old_strategy = birth_death_results[2]

                #play the game
                payoff_mtx = [ [(b-c, b-c), (-c, b)] , [(b, -c), (0,0)] ]
                coop_index={'Cooperate':0, 'Defect':1}
                new_graph = inter.interaction_BD(new_graph, payoff_mtx, delta, noise=0)
                print(nx.get_node_attributes(G, 'fitness'))
                #print(nx.get_node_attributes(G, 'strategy'))
                #print('\n')

                if show_graph:
                    # Creates picture of graph 
                    rep.color_and_draw_graph(new_graph)


                if plotting:
                    if new_strategy != None:
                        # update tallies for each strategy
                        strat_data_dict[new_strategy] += 1
                        strat_data_dict[old_strategy] -= 1

                    # update strategy proportions for each strategy
                    for strat in strat_data_dict:
                        concentrations[strat].append(strat_data_dict[strat]/nx.number_of_nodes(G))

                    #print("Current time data is", time_data)
                    time_data.append(i+1)

                    #print(time_data)
                    #print(final_data)
                    #print("Plotting data at time ", t)
                    #rep.plot_proportion_data(time_data, final_data)

                #print(new_graph.adj)
                graph = new_graph
            #if plotting:
            #    plot_proportion_data(time_data, final_data, saving, graph_type,t, update_name, n, u, d, data_iteration)

            return new_graph, concentrations, plotting

    def trial_with_plot(G, n, d, data_iteration, u, t, graph_type = 'random', update_name = 'BD', plotting = False, show_graph = False, saving = False):
        '''
        INPUTS:     G: networkx graph object with fitness and strategy attributes
                    u: rate of mutation for reproduction
                    t: number of times to have stratgies update

        OUTPUTS:    new_graph: updated networkx graph object where strategies have been updated

        Prints graph at every stage of strategy updating
        Plots how proportions of each strategy change over time
        '''
        if update_name=='BD':
            if show_graph:
                rep.color_and_draw_graph(G)
                #print(nx.get_node_attributes(G, 'strategy'))
                #print("-----------------------------------------------")

            if plotting:
                time_data = [0]
                final_data = {}
                proportions_dict = {}
                strat_data_dict = {}
                # initialize strategy tallies (frequencies) to 0
                for strat in strat_list:
                    strat_data_dict[strat] = 0
                    final_data[strat] = []

                # if a strategy is found in the graph, adjust its tally
                for n in nx.nodes(G):
                    strat_data_dict[G.node[n]['strategy']] += 1
                # final_data maps       strategy ---> freq(strat)/number(nodes)
                for strat in strat_data_dict:
                    final_data[strat].append(strat_data_dict[strat]/nx.number_of_nodes(G))


            for i in range(t):
                # TODO-----------------------------
                # figure out why variables have to be directly input
                # into the birth-death function

                #adjust node strategies 
                birth_death_results = rep.birth_death(G, strat_list, u)
                new_graph = birth_death_results[0]
                new_strategy = birth_death_results[1]
                old_strategy = birth_death_results[2]

                #play the game
                payoff_mtx = [ [(b-c, b-c), (-c, b)] , [(b, -c), (0,0)] ]
                coop_index={'Cooperate':0, 'Defect':1}
                new_graph = inter.interaction_BD(new_graph, payoff_mtx, delta, noise=0)

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
                    #print("Plotting data at time ", t)
                    #rep.plot_proportion_data(time_data, final_data)

                #print(new_graph.adj)
                graph = new_graph

            if plotting:
                plot_proportion_data(time_data, final_data, saving, graph_type,t, update_name, n, u, d, data_iteration)

            #rep.color_and_draw_graph(new_graph)

            return new_graph

def get_histogram_and_concentration_dict(G, strat_list):
    '''
    to make an initial histogram for the strategy frequencies
    and an initial record for the strategy concentrations
    '''
    histo_dict = {}
    conc_dict = {}
    # initialize strategy tallies (frequencies) to 0
    for strat in strat_list:
        histo_dict[strat] = 0
        conc_dict[strat] = []
    # if a strategy is found in the graph, adjust its tally
    for n in nx.nodes(G):
        histo_dict[G.node[n]['strategy']] += 1
    # final_data maps       strategy ---> freq(strat)/number(nodes)
    for strat in histo_dict.keys():
        conc_dict[strat].append(histo_dict[strat]/nx.number_of_nodes(G))
    return histo_dict, conc_dict

def plot_many_trials(G, n, d, data_iteration, u, t, number_trials, the_strat, graph_type = 'random', \
    update_name = 'BD', plotting = True, show_graph = False, saving = False):    
    '''
    matrix in which entry n,t is the concentration 
    of the_strat at time t in trial n
    '''
    #matrix in which entry n,t is the concentration 
    #of the_strat at time t in trial n
    result_matrix=[]
    #run the game for each trial
    for each in range(number_trials):
        this_game=game(G, update_name, t, u, d, plotting, show_graph)
        trial_outcome=this_game.trial(G, n, d, data_iteration, u, t, graph_type, update_name, plotting, show_graph, saving)
        #append record for this trial of the concentrations of the_strat
        result_matrix.append(trial_outcome[1][the_strat])
    #scatter plot
    X=[tictoc+1 for tictoc in range(t)]
    #three lines to plot: average, and pm stdev
    Yavg, Yplus, Yminus=[], [], []
    for tictoc in range(t):
        at_time_t=[trial[tictoc] for trial in result_matrix]
        #average at time t over all the trials
        average=sum(at_time_t)/len(at_time_t)
        stdev=np.std(at_time_t)
        Yavg.append(average)
        Yplus.append(average+stdev)
        Yminus.append(average-stdev)

    #plot the 3 lines
    plt.plot(X, Yavg, color='green', marker='', linestyle = '-')
    plt.plot(X, Yplus, color='red', marker='', linestyle = '-')
    plt.plot(X, Yminus, color='blue', marker='', linestyle = '-')
    
    #change axes ranges
    plt.xlim(0,t)
    plt.ylim(0,1)
    #add title
    plt.title('Relationship between time and proportion of nodes with strategy ' + the_strat + ' in '+str(number_trials)+ ' trials')
    #add x and y labels
    plt.ylabel('Proportion of nodes with strategy ' + the_strat)
    plt.xlabel('Time')

    #show plot
    plt.show()   
    plt.close()     

    #if saving:
    #    print("Attempting to save plot ", data_iteration)
    #    plt.savefig(graph_type + '_' + str(t) + '_' + update_name + '_n=' + str(n) + '_u=' + \
    #       str(u) + '_d=' + str(d) + '_' + 'trial' + str(data_iteration) + '.png')
    #plt.close()

    return None

def plot_proportion_data(time, strat_dict, saving, graph_type, t, update_name, n, u, d, data_iteration):
    for strat in strat_dict:

        # IMPORTANT -- UNDO THIS IF STATEMENT WHEN EXAMINING MORE STRATEGIES
        if strat == 'Cooperate':
            #scatter plot
            X = time
            Y = strat_dict[strat]
            plt.plot(X, Y, color='blue', marker='^', linestyle = '-')

            #change axes ranges
            plt.xlim(0,max(time))
            plt.ylim(0,1)

            #add title
            plt.title('Relationship between time and proportion of nodes with strategy ' + strat)

            #add x and y labels
            plt.ylabel('Proportion of nodes with strategy ' + strat)
            plt.xlabel('Time')

            #show plot
            plt.show()   
            #plt.close()     

            if saving:
                print("Attempting to save plot ", data_iteration)
                plt.savefig(graph_type + '_' + str(t) + '_' + update_name + '_n=' + str(n) + '_u=' \
                    + str(u) + '_d=' + str(d) + '_' + 'trial' + str(data_iteration) + '.png')
            #plt.close()

    return None


'''
def plot_many_tests(time, strat_dict, saving, graph_type, t, update_name, n, u, d, i):
    
    Like plot proportional data, but handles many dictionaries
    from the data of many graphs

    Plots each data a different color/style on the same plot
    
    for strat in strat_dict:
        # IMPORTANT -- UNDO THIS IF STATEMENT WHEN EXAMINING MORE STRATEGIES
        if strat == 'Cooperate':
            #scatter plot
            X = time
            Y = strat_dict[strat]
            plt.plot(X, Y, color='blue', marker='^', linestyle = '-')

            #change axes ranges
            plt.xlim(0,max(time))
            plt.ylim(0,1)

            #add title
            plt.title('Relationship between time and proportion of nodes with strategy ' + strat)

            #add x and y labels
            plt.ylabel('Proportion of nodes with strategy ' + strat)
            plt.xlabel('Time')

            #show plot
            plt.ion()        

            if saving:
                print("Attempting to save plot ", i)
                plt.savefig(graph_type + '_' + str(t) + '_' + update_name + '_n=' + str(n) + \
                '_u=' + str(u) + '_d=' + str(d) + '_' + 'trial' + str(i) + '.png')
            plt.close()

    return None
'''

'''--------------------------------------------
                        TESTS
--------------------------------------------'''

'''--------------------------
        IMPORTANT VARS
--------------------------'''
#strat_list = ['Cooperate', 'Defect', 'Tit_for_tat']
strat_list = ['Cooperate', 'Defect']
u = .2

b = 8
c = 1
delta = 20

n=10
d=3
graph_type = 'dumbbell'
update_name = 'BD'
time_length = 3
number_trials=10

n_lattice = 50
m_lattice = 50

start_prop_cooperators = .2



'''------------
TYPES OF GRAPHS
------------'''

#Lattice
#G=init.generate_lattice(n_lattice, m_lattice)

#Complete graph
G = init.generate_graph(n, graph_type)

#Random regular graph
#G = init.generate_graph(n, type = 'random', d)

#Erdos-Reyni
#G = init.generate_graph(n, type = 'random', d, 40)

'''--------------
LABELS
--------------'''

init.label_dumbbell_birth_death(G, strat_list)
#init.label_birth_death(G, strat_list, start_prop_cooperators)
#init.label_BD_according_to_one_dim(G, strat_list, n_lattice)
rep.color_and_draw_graph(G)

'''-------------
TIMESTEP
--------------'''
'''
for data_iteration in range(1):
    game.trial_with_plot(G, n, d, data_iteration, u, time_length, graph_type, \
        update_name= 'BD', plotting = True, show_graph = True, saving = False)
'''

data_iteration=[]
plot_many_trials(G, n, d, data_iteration, u, time_length, number_trials, 'Cooperate', graph_type, 'BD', plotting=True, show_graph=False, saving=False)







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



