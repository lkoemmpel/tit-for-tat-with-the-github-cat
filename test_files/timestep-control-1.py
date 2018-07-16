'''-------------------
        PACKAGES
-------------------'''
import random
import networkx as nx
import matplotlib.pyplot as plt
import math 
import numpy as np 
import pylab

from matplotlib import colors as mcolors

'''-------------------
        IMPORTS
-------------------'''
import graph_initialization as init
import reproduction as rep
import interaction as inter
import display as dis

from matplotlib.pyplot import pause

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


names_to_functions={'BD': rep.birth_death, 'DB': rep.death_birth}

class game():
    def __init__(self, graph, name, t, u=0, delta=0, plotting = False, show_graph = False, saving=False, color_fitness=False):
        self.graph=graph
        #name of the game
        self.name=name
        #mutation rate
        self.u = u
        #strength of selection
        self.delta = delta
        #time of a trial
        self.t = t
        #are we going to plot?
        self.plotting = plotting
        #do we want to show graph?
        self.show_graph = show_graph
        #do we want to save images?
        self.saving=saving
        #do we want to use a heat map for fitness?
        self.color_fitness=color_fitness

    def trial(self, pos, num_rep, graph_type = 'random', num_of_trial=None):
        '''
        INPUTS:     G: networkx graph object with fitness and strategy attributes
                    u: rate of mutation for reproduction
                    t: number of times to have stratgies update
        OUTPUTS:    new_graph: updated networkx graph object where strategies have been updated
        Prints graph at every stage of strategy updating
        Plots how proportions of each strategy change over time
        '''

        G=self.graph
        u=self.u
        t=self.t


        if type(self.name)==str:
        #if update_name=='BD':

            if self.show_graph:
                dis.color_fitness_and_draw_graph(G, pos, None, num_of_trial, 0)
                #print(nx.get_node_attributes(G, 'strategy'))
                #print("-----------------------------------------------")

            time_data=[]
            strat_data_dict, concentrations= get_histogram_and_concentration_dict(G, strat_list)
            # strat_data_dict maps      strategy ---> freq(strat)
            # concentrations maps       strategy ---> matrix 
            #                                         entry n,t : freq(strat)/number(nodes)
            #                                         at trial n, time t

            for i in range(t):
                print("Running time ", t)
                #------------
                #REPRODUCTION
                #------------ 
                birth_death_results = names_to_functions[update_name](G, strat_list, u, num_rep)
                #naming the results from rep
                new_graph = birth_death_results[0]
                new_strategies=birth_death_results[1]
                old_strategies = birth_death_results[2]
                reproducing_nodes = birth_death_results[3]

                #------------
                #INTERACTION
                #------------
                payoff_mtx = [ [(b-c, b-c), (-c, b)] , [(b, -c), (0,0)] ]
                coop_index={'Cooperate':0, 'Defect':1}
                new_graph = inter.interaction_depends_fitness(new_graph, payoff_mtx, delta, noise=0.5)
                #print(nx.get_node_attributes(G, 'fitness'))
                #print(nx.get_node_attributes(G, 'strategy'))
                #print('\n')

                if self.show_graph:
                    if i%1 == 0:
                        if self.color_fitness:
                            dis.color_fitness_and_draw_graph(new_graph, pos, reproducing_nodes, num_of_trial, i+1)
                        else:
                            # Creates picture of graph 
                            dis.color_and_draw_graph(new_graph)

                for index in range(len(old_strategies)):
                    new=new_strategies[index]
                    old=old_strategies[index]
                    strat_data_dict[new]+=1
                    strat_data_dict[old]-=1

                # if new_strategy != None:
                #     # update tallies for each strategy
                #     strat_data_dict[new_strategy] += 1
                #     strat_data_dict[old_strategy] -= 1

                # update strategy proportions for each strategy
                for strat in strat_data_dict:
                    concentrations[strat].append(strat_data_dict[strat]/nx.number_of_nodes(G))

                #print("Current time data is", time_data)
                time_data.append(i+1)

                #print(time_data)
                #print(final_data)
                #print("Plotting data at time ", t)
                #plot_proportion_data(time_data, final_data)

                graph = new_graph

            #if self.plotting:
            #    plot_proportion_data(time_data, final_data, saving, graph_type,t, update_name, n, u, d, data_iteration)

            return new_graph, concentrations, self.plotting

    def lattice_density_and_payoff_trial(self, pos, num_rep, graph_type = 'random'):
        '''
        INPUTS:     G: networkx graph object with fitness and strategy attributes
                    u: rate of mutation for reproduction
                    t: number of times to have stratgies update
        OUTPUTS:    new_graph: updated networkx graph object where strategies have been updated
        Prints graph at every stage of strategy updating
        Plots how proportions of each strategy change over time
        '''

        G=self.graph
        u=self.u
        t=self.t


        if type(self.name)==str:
        #if update_name=='BD':

            if self.show_graph:
                dis.color_fitness_and_draw_graph(G, pos)
                #print(nx.get_node_attributes(G, 'strategy'))
                #print("-----------------------------------------------")

            if self.plotting:
                time_data=[]
                strat_data_dict, concentrations= get_histogram_and_concentration_dict(G, strat_list)
                # strat_data_dict maps      strategy ---> freq(strat)
                # concentrations maps       strategy ---> matrix 
                #                                         entry n,t : freq(strat)/number(nodes)
                #                                         at trial n, time t

            for i in range(t):
                #------------
                #REPRODUCTION
                #------------ 
                birth_death_results = names_to_functions[update_name](G, strat_list, u, num_rep)
                #naming the results from rep
                new_graph = birth_death_results[0]
                new_strategies=birth_death_results[1]
                old_strategies = birth_death_results[2]
                reproducing_nodes = birth_death_results[3]

                #new_strategy = birth_death_results[1]
                #old_strategy = birth_death_results[2]

                #------------
                #INTERACTION
                #------------
                payoff_mtx = [ [(b-c, b-c), (-c, b)] , [(b, -c), (0,0)] ]
                coop_index={'Cooperate':0, 'Defect':1}
                new_graph = inter.interaction_BD(new_graph, payoff_mtx, delta, noise=0)
                #print(nx.get_node_attributes(G, 'fitness'))
                #print(nx.get_node_attributes(G, 'strategy'))
                #print('\n')

                if self.show_graph:
                    if i%1 == 0:
                        if self.color_fitness:
                            dis.color_fitness_and_draw_graph(new_graph, pos, reproducing_nodes)
                        else:
                            # Creates picture of graph 
                            dis.color_and_draw_graph(new_graph)

                if self.plotting:
                    for index in range(len(old_strategies)):
                        new=new_strategies[index]
                        old=old_strategies[index]
                        strat_data_dict[new]+=1
                        strat_data_dict[old]-=1

                    # if new_strategy != None:
                    #     # update tallies for each strategy
                    #     strat_data_dict[new_strategy] += 1
                    #     strat_data_dict[old_strategy] -= 1

                    # update strategy proportions for each strategy
                    for strat in strat_data_dict:
                        concentrations[strat].append(strat_data_dict[strat]/nx.number_of_nodes(G))

                    #print("Current time data is", time_data)
                    time_data.append(i+1)

                    #print(time_data)
                    #print(final_data)
                    #print("Plotting data at time ", t)
                    #plot_proportion_data(time_data, final_data)

                graph = new_graph

            #if self.plotting:
            #    plot_proportion_data(time_data, final_data, saving, graph_type,t, update_name, n, u, d, data_iteration)

            return new_graph, concentrations, self.plotting

    def trial_with_plot(G, n, d, data_iteration, u, t, num_rep = 1, graph_type = 'random'):
        '''
        INPUTS:     G: networkx graph object with fitness and strategy attributes
                    u: rate of mutation for reproduction
                    t: number of times to have stratgies update
        OUTPUTS:    new_graph: updated networkx graph object where strategies have been updated
        Prints graph at every stage of strategy updating
        Plots how proportions of each strategy change over time
        '''
        G=self.graph
        n=len(list(nx.nodes(G)))

        if self.name=='BD':
            if self.show_graph:
                rep.color_and_draw_graph(G)
                #print(nx.get_node_attributes(G, 'strategy'))
                #print("-----------------------------------------------")

            if self.plotting:
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
                birth_death_results = rep.birth_death(G, strat_list, u, num_rep)
                new_graph = birth_death_results[0]
                new_strategy = birth_death_results[1]
                old_strategy = birth_death_results[2]

                #play the game
                payoff_mtx = [ [(b-c, b-c), (-c, b)] , [(b, -c), (0,0)] ]
                coop_index={'Cooperate':0, 'Defect':1}
                new_graph = inter.interaction_BD(new_graph, payoff_mtx, delta, noise=0)

                #print(nx.get_node_attributes(G, 'strategy'))
                #print('\n')

                if self.show_graph:
                    if i%100 ==0:
                        # Creates picture of graph 
                        dis.color_fitness_and_draw_graph(new_graph)


                if self.plotting:
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

            if self.plotting:
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
        histo_dict[nx.get_node_attributes(G, 'strategy')[n]]+=1
        #G.node(n)['strategy']] += 1
    # final_data maps       strategy ---> freq(strat)/number(nodes)
    for strat in histo_dict.keys():
        conc_dict[strat].append(histo_dict[strat]/nx.number_of_nodes(G))
    return histo_dict, conc_dict

def plot_many_trials(parameters, graph_type, u, t, number_trials, the_strat, num_rep, \
    rho = None, update_name = 'BD', plotting = True, show_graph = False, saving = False, color_fitness=False):    

    #matrix in which entry n,t is the concentration 
    #of the_strat at time t in trial n
    result_matrix=[]
    #run the game for each trial
    for each in range(number_trials):
        print('\n')
        print("Running trial ", each)
        #print("Evaluating trial ", each)
        graph=init.generate_graph(parameters, graph_type)

        if rho != None:
            # Remove lattice nodes until only rho percent remain
            sparse_graph = graph.copy()
            for n in graph.nodes():
                #indicator is random.uniform(0,1)
                if random.uniform(0,1) > rho:
                    if nx.number_of_nodes(sparse_graph) != 1:
                        #this node should be deleted
                        sparse_graph.remove_node(n)
            graph = sparse_graph


        init.label_birth_death(graph, strat_list, start_prop_cooperators)
        #init.label_BD_according_to_one_dim(graph, strat_list, parameters[1])        

        this_game=game(graph, update_name, t, u, d, plotting, show_graph, saving, color_fitness)


        if graph_type == 'triangular_lattice':
            pos = dict( (n, n) for n in graph.nodes() )
        else:
            pos = nx.spring_layout(graph)

        trial_outcome = this_game.trial(pos, num_rep, graph_type, each+1)
        

        #trial_outcome=this_game.trial(graph, u, t, nx.spring_layout(graph, 1/n**.2), \
        #    graph_type, update_name, plotting, show_graph, saving, color_fitness)
        #append record for this trial of the concentrations of the_strat
        
        result_matrix.append(trial_outcome[1][the_strat])


    #scatter plot X axis! 
    X=[tictoc for tictoc in range(t)]
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

    if plotting:
        #plot the 3 lines
        plt.plot(X, Yavg, color='green', marker='', linestyle = '-')
        plt.plot(X, Yplus, color='red', marker='', linestyle = '-')
        plt.plot(X, Yminus, color='blue', marker='', linestyle = '-')
        
        #change axes ranges
        plt.xlim(0,t-1)
        plt.ylim(0,1)
        #add title
        plt.title('Relationship between time and proportion of nodes with \n strategy ' + the_strat + ' in '+str(number_trials)+ ' trials')
        #add x and y labels
        plt.ylabel('Proportion of nodes with strategy ' + the_strat)
        plt.xlabel('Time')

        #show plot
        plt.show()   
        print("Attempting to show plot -----------------")
        pause(60)
    #plt.close()     

    if saving:
    #    print("Attempting to save plot ", data_iteration)+1
        plt.savefig(graph_type + '_' + str(t) + '_' + update_name + '_n=' + str(n) + '_u=' + \
            str(u) + '_d=' + str(d) + '_' + 'trial' + str(data_iteration) + '.png')
    #plt.close()

    #last_5=Yavg[-5:]
    #return sum(last_5)/len(last_5)
    return Yavg[-1]

def plot_lattice_density_and_payoff(parameters, graph_type, u, t, max_b, the_strat, \
    update_name = 'BD', plotting = True, show_graph = False, saving = False, color_fitness = True):    

    #matrix in which entry n,rho is the concentration 
    #of the_strat at population density rho when b=n

    #run the game for each value of b

    b_increments = np.arange(0, max_b, 0.01)
    print(len(b_increments))

    #Currently there are 148 possible colors
    remaining_colors = list(mcolors.CSS4_COLORS.keys())
    #remaining_colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for b in b_increments:    

        #initialize lists for this b value's graph
        rho_values = []
        prop_coop_values = []

        rho_increments = np.arange(0, 1.01, 0.01)
        for rho in rho_increments:
            rho_values.append(rho)
            prop_coop_values.append(plot_many_trials(parameters, graph_type, u, t, number_trials, the_strat, num_rep, \
                rho, update_name = 'BD', plotting = False, show_graph = False, saving = False, color_fitness=False))

        X = rho_values
        Y = prop_coop_values
        this_color=remaining_colors.pop()
        plt.plot(X, Y, color=this_color, marker='', linestyle = '-', label= 'b= '+str(b))

        pylab.legend(loc='lower left')
    
        #change axes ranges
        plt.xlim(0,1)
        plt.ylim(0,1)
        #add title
        plt.title('Relationship between population density and long-term proportion \n of nodes with strategy ' + the_strat + ' in '+str(number_trials)+ ' trials')
        #add x and y labels
        plt.ylabel('Proportion of nodes with strategy ' + the_strat)
        plt.xlabel('Population density')

    if plotting:
        #show plot
        plt.show()   
        #pause(600)

    if saving:
        print("Attempting to save plot ")
        plt.savefig(graph_type + '_' + \
                    update_name + '_' + \
                    'n=' + str(n) + '_' + \
                    'm=' + str(m) + '_' + \
                    'u=' + str(u) + '_' + \
                    'prop_coop=' + str(start_prop_cooperators) + '_' + \
                    str(number_trials) + 'trials' + '_' + \
                    str(t) + 'timesteps' + '_' + \
                    'b_over_c=' + str(b) + '.png')
    return None

def plot_proportion_data(time, strat_dict, saving, graph_type, t, update_name, n, u, d, data_iteration):
    for strat in strat_dict:

        # IMPORTANT -- UNDO THIS IF STATEMENT WHEN EXAMINING MORE STRATEGIES
        if strat == 'Cooperate':
            #scatter plot
            X = time
            Y = strat_dict[strat]
            plt.plot(X, Y, color='blue', marker='', linestyle = '-')

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

def plot_for_different_b_over_c(parameters, graph_type, u, t, number_trials, the_strat, num_rep, \
    update_name = 'BD', plotting = True, show_graph = False, saving = False, color_fitness=False):
    return
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
u = .4

b = 3
c = 1
delta = .2

n = 5
m = 5

d = 6
graph_type = 'dumbell_multiple'

update_name = 'BD'
#list of parameters that will be used to build graph
parameters = [n,m]


t = 100


number_trials=5


n_lattice = 50
m_lattice = 50

start_prop_cooperators = .5

#Number of nodes to reproduce at each timestep 
num_rep = 5

'''------------
TYPES OF GRAPHS
------------'''

#Erdos-Renyi graph
#G=init.generate_graph([d,m], graph_type)

#Complete/Dumbbell graph
#G = init.generate_graph([n], graph_type)

#Multiple dumbell
#G=init.generate_dumbell_multiple_cliques(10,5,1)

#Multiple dumbell
#G1=init.generate_graph([5,4,2], graph_type)

#Basic dumbell
#G2=init.generate_graph([20],'dumbell')

#Random regular graph
#G = init.generate_graph([n,d], graph_type)

#Erdos-Reyni
#G = init.generate_graph([n,d,40], graph_type)

'''--------------
LABELS
--------------'''

#init.label_birth_death(G, strat_list, start_prop_cooperators)
#init.label_BD_according_to_one_dim(G, strat_list, d)
#init.label_dumbbell_birth_death(G, strat_list)
#init.label_dumbell_multiple_cliques(G1, {0,1,3})
#dis.color_fitness_and_draw_graph(G, nx.spring_layout(G))


'''-------------
TIMESTEP
--------------'''

#1                      Test for trial_with_plot

# for data_iteration in range(5):
#     #init.label_birth_death(G, strat_list, start_prop_cooperators)
#     G=init.generate_dumbell_multiple_cliques(10,5,1)
#     init.label_birth_death(G, strat_list, start_prop_cooperators)
#     #rep.color_and_draw_graph(G)

#     game.trial_with_plot(G, n, d, data_iteration, u, time_length, graph_type, \
#         update_name= 'BD', plotting = True, show_graph = False, saving = False)


#2                      Test for plot_many_trials
c=1
b=1
max_b = 2

#prop_increments = np.arange(.3, 1.1, 0.1)

#start_prop_cooperators=.1

#for start_prop_cooperators in prop_increments:
#plot_lattice_density_and_payoff(parameters, graph_type, u, t, max_b, 'Cooperate', update_name = 'BD', \
#    plotting = True, show_graph = False, saving = True, color_fitness = True)


parameters=[30,2,4]
plot_many_trials(parameters, graph_type, u, t, number_trials, 'Cooperate', num_rep, None, 'BD', plotting=True, show_graph=True, saving=True, color_fitness=True)




