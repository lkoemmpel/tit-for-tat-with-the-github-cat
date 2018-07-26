'''-------------------
        PACKAGES
-------------------'''
import random
from random import randint 
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

light_colors = ['whitesmoke','bisque','floralwhite','lightgoldenrodyellow','aliceblue','ghostwhite',\
                'ivory','lemonchiffon','seashell','lightgray','lightgrey','white','beige','honeydew',\
                'azure','lavender','gainsboro','snow','linen','antiquewhite','papayawhip','oldlace','cornsilk',\
                'lightyellow','mintcream','lightcyan','lavenderblush', 'moccasin', 'peachpuff', 'navajowhite', \
                'blanchedalmond', 'wheat', 'khaki', 'palegoldenrod', 'yellow']

'''--------------------------
        HELPERS
--------------------------'''

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

def get_props_cliques (G):
    '''
    G is a multiple dumbell
    '''
    num_cliques=0
    for n in G.nodes():
        if G.node[n]['coord'][0][0] == G.node[n]['coord'][0][1]:
            num_cliques = max(G.node[n]['coord'][0][0]+1, num_cliques)
    #print(num_cliques)
    counts = [ 0 for k in range(num_cliques) ]
    sizes = [ 0 for k in range(num_cliques) ]

    for n in G.nodes():
        if G.node[n]['coord'][0][0] == G.node[n]['coord'][0][1]:
            k = G.node[n]['coord'][0][0]
            sizes[k]+=1
            if G.node[n]['strategy'] == 'Cooperate':
                counts[k]+=1

    #MAKE A LIST:       props[k]=proportion at clique k
    props = [ counts[k]/sizes[k] for k in range(num_cliques) ]
    return props

def info(G):
    fits={}
    for n in G.nodes():
        fits[n]=G.node[n]['fitness']
    return fits


'''----------------------------------
        THE ONE AND ONLY (code)
----------------------------------'''


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

    def trial(self, pos, num_rep, noise=0, graph_type = 'random', num_of_trial=None):
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

            D_list=[]
            for i in range(t):
                D_list.append(D(G,delta,b,c))
                '''
                print('********')
                print('TIME '+ str(i))
                print('********')
                '''
                #print("Running time ", t)
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

                new_graph = inter.interaction_BD(new_graph, payoff_mtx, delta, noise)

                #print(nx.get_node_attributes(G, 'fitness'))
                #print(nx.get_node_attributes(G, 'strategy'))
                #print('\n')

                if self.show_graph:
                    if i%1 == 0:
                        if self.color_fitness:
                            dis.color_fitness_and_draw_graph(new_graph, pos, reproducing_nodes, num_of_trial, i+1)
                        else:
                            # Creates picture of graph 
                            print('------------------------Fitness variable was false')
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

                G = new_graph


            #if self.plotting:
            #    plot_proportion_data(time_data, final_data, saving, graph_type,t, update_name, n, u, d, data_iteration)

            return new_graph, concentrations, self.plotting, D_list

    def trial_multidumbell(self, pos, num_rep, noise, b, c, graph_type = 'random', num_of_trial=None):
        '''
        INPUTS:     G: networkx graph object with fitness and strategy attributes
                    u: rate of mutation for reproduction
                    t: number of times to have stratgies update
        OUTPUTS:    new_graph: updated networkx graph object where strategies have been updated
        Prints graph at every stage of strategy updating
        Plots how proportions of each strategy change over time
        '''
        graph=self.graph
        u=self.u
        t=self.t

        if type(self.name)==str:

            if self.show_graph:
                dis.color_fitness_and_draw_graph(graph, pos, None, num_of_trial, 0)
                #print(nx.get_node_attributes(G, 'strategy'))
                #print("-----------------------------------------------")
            
            time_data=[]
            strat_data_dict, concentrations= get_histogram_and_concentration_dict(graph, strat_list)
            # strat_data_dict maps      strategy ---> freq(strat)
            # concentrations maps       strategy ---> matrix 
            #                                         entry n,t : freq(strat)/number(nodes)
            #                                         at trial n, time t
            props=[]
            D_list = []
            for i in range(t):
                D_list.append(D(graph,delta,b,c))

                dictio=info(graph)
                '''
                print('********')
                print('TIME '+ str(i))
                print('********')
                '''
                #print("Running time ", t)
                #----------------------
                #PROPORTIONS PER CLIQUE
                #---------------------- 
                props.append(get_props_cliques(graph))
                #------------
                #REPRODUCTION
                #------------ 
                birth_death_results = names_to_functions[update_name](graph, strat_list, u, num_rep)
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
                new_graph = inter.interaction_BD(new_graph, payoff_mtx, delta, noise)
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
                    concentrations[strat].append(strat_data_dict[strat]/nx.number_of_nodes(graph))

                #print("Current time data is", time_data)
                time_data.append(i+1)

                #print(time_data)
                #print(final_data)
                #print("Plotting data at time ", t)
                #plot_proportion_data(time_data, final_data)

                graph = new_graph

            #if self.plotting:
            #    plot_proportion_data(time_data, final_data, saving, graph_type,t, update_name, n, u, d, data_iteration)

            return new_graph, concentrations, self.plotting, props, D_list

    def trial_utkovski(self, pos, noise, b, c, num_of_trial=None):
        '''
        INPUTS:     G: networkx graph object with fitness and strategy attributes
                    u: rate of mutation for reproduction
                    t: number of times to have stratgies update
        OUTPUTS:    new_graph: updated networkx graph object where coop states have been updated
        Prints graph at every stage of utkovski game
        Plots how many helping actions occured over time
        '''

        G=self.graph
        u=self.u
        t=self.t

        if type(self.name)==str:

            time_data=[]
            helpers=[]
            help_accross_time=[]

            if self.show_graph:
                dis.heat_map_utkovski(G, pos, helpers, num_of_trial, 0)

            for i in range(t):
                print('********')
                print('TIME '+ str(i))
                print('********')
                #print("Running time ", i)
                #------------
                #INTERACTION UTKOVSKI TYPE
                #------------ 
                set_nodes = set(G.nodes())
                outcome = inter.general_reciprocity_simple(G, set_nodes, b, c, False)
                new_graph = outcome[0]
                helpers = outcome[1]
                num_help_actions = outcome[2]
                help_accross_time.append(num_help_actions)

                if self.show_graph:
                    if i%1 == 0:
                        if self.color_fitness:
                            dis.heat_map_utkovski(new_graph, pos, helpers, num_of_trial+1, i+1)
                        else:
                            # Creates picture of graph C/D 
                            print('------------------------Fitness variable was false')
                            dis.color_and_draw_graph(new_graph)

                #print("Current time data is", time_data)
                time_data.append(i+1)

                #print(time_data)
                G = new_graph

            #if self.plotting:
            #    plot_proportion_data(time_data, final_data, saving, graph_type,t, update_name, n, u, d, data_iteration)

            return G, help_accross_time

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
                            print("-----------------fitness is not going to be colored")
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

def plot_many_trials(parameters, graph_type, u, delta, noise, t, number_trials, the_strat, num_rep, \
    rho = None, update_name = 'BD', plotting = True, show_graph = False, saving = False, color_fitness=False):    

    #matrix in which entry n,t is the concentration 
    #of the_strat at time t in trial n
    result_matrix=[]
    #run the game for each trial
    for each in range(number_trials):
        print('\n')
        print("Running trial ", each)
        #print("Evaluating trial ", each)
        graph=init.generate_weighted(parameters, graph_type)

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

        #LABEL BIRTH DEATH

        print("We are labelling allen")
        init.label_allen(graph, strat_list, start_prop_cooperators)
        #LABEL FOR A LATTICE WITH ONE SLICE OF DEFECTORS
        #init.label_BD_according_to_one_dim(graph, strat_list, parameters[1]) 
        #LABEL MULTIPLE CLIQUES 
        #init.label_dumbell_multiple_cliques(graph, strat_list, {0: 0.2, 1:0.9, 2:0, 3:0.1, 4:1})       


        this_game=game(graph, update_name, t, u, delta, plotting, show_graph, saving, color_fitness)


        if graph_type == 'triangular_lattice':
            pos = dict( (n, n) for n in graph.nodes() )
        else:
            pos = nx.spring_layout(graph)

        trial_outcome = this_game.trial(pos, num_rep, noise, graph_type, each+1)
        

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

    the_avg = sum(Yavg)/len(Yavg)
    AVG=[the_avg for tictoc in range(t)]

    if plotting:
        #plot the 3 lines
        plt.plot(X, Yavg, color='green', marker='', linestyle = '-')
        plt.plot(X, Yplus, color='red', marker='', linestyle = '-')
        plt.plot(X, Yminus, color='blue', marker='', linestyle = '-')
        plt.plot(X, AVG, color='orange', marker='', linestyle= '-')

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
        pause(4)   
        #print("Attempting to show plot -----------------")
        #pause(60)
    #plt.close()     

    if saving:
        file_id = 0
        #randint(10**5, 10**6 - 1)
    #   print("Attempting to save plot ", data_iteration)+1
        if graph_type=='dumbell_multiple':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'size_dumbell=' + str(parameters[0]) + '_' + \
                'num_dumbell=' + str(parameters[1]) + '_' + \
                'size_path=' + str(parameters[2]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='dumbell':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'n=' + str(parameters[0]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type == 'rich_club':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'size_club=' + str(parameters[0]) + '_' + \
                'size_periphery=' + str(parameters[1]) + '_' + \
                'prob_rp=' + str(parameters[2]) + '_' + \
                'prob_rr=' + str(parameters[3]) + '_' + \
                'prob_pp=' + str(parameters[4]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='complete' or 'hypercube':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'num_nodes=' + str(parameters[0]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + \
                str(file_id) + '.png')
        elif graph_type=='triangular_lattice':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'n_dim=' + str(parameters[0]) + '_' + \
                'm_dim=' + str(parameters[1]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='random':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'num_nodes=' + str(parameters[0]) + '_' + \
                'ave_degree=' + str(parameters[1]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + \
                str(file_id) + '.png')

    return Yavg[-1], graph 

def plot_multiple_dumbell_each_clique(parameters, graph_type, u, b, c, delta, noise, t, number_trials, the_strat, num_rep, this_color, \
    rho = None, update_name = 'BD', plotting = True, show_graph = False, saving = False, color_fitness=False):    

    #matrix in which entry n,t is the concentration 
    #of the_strat at time t in trial n
    clique_data = {}
    result_matrix = []
    result_matrix_D = []
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

        #LABEL BIRTH DEATH
        #init.label_birth_death_precise_prop(graph, strat_list, start_prop_cooperators)
        #LABEL FOR A LATTICE WITH ONE SLICE OF DEFECTORS
        #init.label_BD_according_to_one_dim(graph, strat_list, parameters[1]) 
        #LABEL MULTIPLE CLIQUES 
        try:
            cliques_to_proportions = parameters[3]
        except:
            cliques_to_proportions = parameters[2]

        init.label_dumbell_multiple_cliques_allen(graph, b, c, strat_list, cliques_to_proportions)       


        this_game=game(graph, update_name, t, u, delta, plotting, show_graph, saving, color_fitness)

        pos = nx.spring_layout(graph)

        trial_outcome = this_game.trial_multidumbell(pos, num_rep, noise, b, c, graph_type, each+1)
        

        #trial_outcome=this_game.trial(graph, u, t, nx.spring_layout(graph, 1/n**.2), \
        #    graph_type, update_name, plotting, show_graph, saving, color_fitness)
        #append record for this trial of the concentrations of the_strat

        props = trial_outcome[3]
        result_matrix.append(props)
        D=trial_outcome[4]
        result_matrix_D.append(D)

    ########################################################
    ############ P L O T - C O O P - P R O P ###############
    ########################################################
    # create plot where each line shows prop cooperators over time
    # for a single clique 
    X=[tictoc for tictoc in range(t)]

    #Currently there are 148 possible colors
    '''
    remaining_colors = list(mcolors.CSS4_COLORS.keys())
    for color in light_colors:
        #print("Trying to remove ", color)
        remaining_colors.remove(color)
    '''

    Y_data = {}
    Yplus_data = {}
    Yminus_data = {}
    num_cliques = len(props[0])
    for clique in range(num_cliques):
        Yavg = []
        Yplus = []
        Yminus = []
        for tictoc in range(t):
            at_time_t=[trial[tictoc][clique] for trial in result_matrix]
            #average at time t and given clique over all the trials

            average=sum(at_time_t)/len(at_time_t)
            Yavg.append(average)

            stdev=np.std(at_time_t)
            Yplus.append(average+stdev)
            Yminus.append(average-stdev)

        Y_data[clique] = Yavg
        Yplus_data[clique] = Yplus
        Yminus_data[clique] = Yminus

    if plotting:
        #plot the 3 lines
        linestyles = [':', '-.', '--', '-']
        for clique in Y_data:
            ###### WARNING THIS ONLY WORKS FOR \leq 4 MULTIPLE DUMBELLS ######
            line_type = linestyles[clique]
            ##################################################################

            #this_color=remaining_colors.pop()
            plt.figure(1)
            plt.plot(X, Y_data[clique], color=this_color, marker='', linestyle = line_type, label='clique '+str(clique))
            #plt.plot(X, Yplus_data[clique], color=this_color, marker='', linestyle = ':', label='clique '+str(clique))
            #plt.plot(X, Yminus_data[clique], color=this_color, marker='', linestyle = ':', label='clique '+str(clique))

        #pylab.legend(loc='lower left')

        
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
        #pause(3)  

    if saving:
        file_id = randint(10**5, 10**6 - 1)
        #print("Attempting to save plot ", data_iteration)+1
        plt.savefig(graph_type + '_' + \
            update_name + '_' + \
            'u=' + str(u) + '_' + \
            'delta=' + str(delta) + '_' + \
            'b=' + str(b) + '_' + \
            'c=' + str(c) + '_' + \
            'noise=' + str(noise) + '_' + \
            'size_dumbell=' + str(parameters[0]) + '_' + \
            'num_dumbell=' + str(parameters[1]) + '_' + \
            'size_path=' + str(parameters[2]) + '_' + \
            'prop_coop=' + str(start_prop_cooperators) + '_' + \
            str(number_trials) + 'trials' + '_' + \
            str(t) + 'timesteps' + '_' + \
            str(file_id) + '.png')
    

    ########################################################
    #################### P L O T - D #######################
    ########################################################
    #scatter plot X axis! 
    X=[tictoc for tictoc in range(t)]
    #three lines to plot: average, and pm stdev
    
    Yavg = []
    for tictoc in range(t):
        L=[trial[tictoc] for trial in result_matrix_D]
        Yavg.append(sum(L)/len(L)) 
    
    if plotting:
        plt.figure(2)
        plt.plot(X, Yavg, color=this_color, marker='', linestyle = '-')

        #change axes ranges
        plt.xlim(0,t-1)
        plt.ylim(-.25,.25)
        #add title
        plt.title('Relationship between time and D(s) in '+str(number_trials)+ ' trials')
        #add x and y labels
        plt.ylabel('D(s)')
        plt.xlabel('Time')

        #show plot
        plt.show()
        pause(1)   
        #print("Attempting to show plot -----------------")
        #pause(60)
    #plt.close()     

    if saving:
        file_id = 0
        #randint(10**5, 10**6 - 1)
        #print("Attempting to save plot ", data_iteration)+1
        if graph_type=='dumbell_multiple':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'size_dumbell=' + str(parameters[0]) + '_' + \
                'num_dumbell=' + str(parameters[1]) + '_' + \
                'size_path=' + str(parameters[2]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='dumbell':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'n=' + str(parameters[0]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type == 'rich_club':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'size_club=' + str(parameters[0]) + '_' + \
                'size_periphery=' + str(parameters[1]) + '_' + \
                'prob_rp=' + str(parameters[2]) + '_' + \
                'prob_rr=' + str(parameters[3]) + '_' + \
                'prob_pp=' + str(parameters[4]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='complete' or 'hypercube':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'num_nodes=' + str(parameters[0]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + \
                str(file_id) + '.png')
        elif graph_type=='triangular_lattice':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'n_dim=' + str(parameters[0]) + '_' + \
                'm_dim=' + str(parameters[1]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='random':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'num_nodes=' + str(parameters[0]) + '_' + \
                'ave_degree=' + str(parameters[1]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + \
                str(file_id) + '.png')

    return None

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

def plot_many_trials_utkovski(parameters, graph_type, strat_list, start_prop_cooperators, u, this_lambda, kappa, noise, t, number_trials, \
    rho=None, plotting = True, show_graph = False, saving = False, color_fitness=False):    

    #matrix in which entry n,t is the concentration 
    #of the_strat at time t in trial n
    help_matrix=[]
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

        #LABEL UTKOVSKI
        init.label_utkovski(graph, strat_list, start_prop_cooperators)

        this_game=game(graph, update_name, t, u, delta, plotting, show_graph, saving, color_fitness)
        
        if graph_type == 'triangular_lattice':
            pos = dict( (n, n) for n in graph.nodes() )
        else:
            pos = nx.spring_layout(graph)

        trial_outcome = this_game.trial_utkovski(pos, noise, b, c, each)
        help_matrix.append(trial_outcome[1])


    #scatter plot X axis! 
    X=[tictoc for tictoc in range(t)]
    #three lines to plot: average, and pm stdev
    Yavg, Yplus, Yminus=[], [], []
    for tictoc in range(t):
        at_time_t=[trial[tictoc] for trial in help_matrix]
        #average at time t over all the trials
        average=sum(at_time_t)/len(at_time_t)
        stdev=np.std(at_time_t)
        Yavg.append(average)
        Yplus.append(average+stdev)
        Yminus.append(average-stdev)

    the_avg = sum(Yavg)/len(Yavg)
    AVG=[the_avg for tictoc in range(t)]

    if plotting:
        #plot the 3 lines
        plt.plot(X, Yavg, color='green', marker='', linestyle = '-')
        plt.plot(X, Yplus, color='red', marker='', linestyle = '-')
        plt.plot(X, Yminus, color='blue', marker='', linestyle = '-')
        plt.plot(X, AVG, color='orange', marker='', linestyle= '-')

        #change axes ranges
        plt.xlim(0,t-1)
        plt.ylim(0,len(graph.nodes()))
        #add title
        plt.title('Relationship between time and number of helping nodes in \n Utkovski Model, in '+str(number_trials)+ ' trials')
        #add x and y labels
        plt.ylabel('Averaged number of helps accross trials')
        plt.xlabel('Time')

        #show plot
        plt.show()
        pause(20)   
        #print("Attempting to show plot -----------------")
        #pause(60)
    #plt.close()     

    if saving:
        file_id = 0
        #randint(10**5, 10**6 - 1)
    #   print("Attempting to save plot ", data_iteration)+1
        if graph_type=='dumbell_multiple':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'size_dumbell=' + str(parameters[0]) + '_' + \
                'num_dumbell=' + str(parameters[1]) + '_' + \
                'size_path=' + str(parameters[2]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='dumbell':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'n=' + str(parameters[0]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type == 'rich_club':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'size_club=' + str(parameters[0]) + '_' + \
                'size_periphery=' + str(parameters[1]) + '_' + \
                'prob_rp=' + str(parameters[2]) + '_' + \
                'prob_rr=' + str(parameters[3]) + '_' + \
                'prob_pp=' + str(parameters[4]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='complete' or 'hypercube':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'num_nodes=' + str(parameters[0]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + \
                str(file_id) + '.png')
        elif graph_type=='triangular_lattice':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'n_dim=' + str(parameters[0]) + '_' + \
                'm_dim=' + str(parameters[1]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='random':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'num_nodes=' + str(parameters[0]) + '_' + \
                'ave_degree=' + str(parameters[1]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + \
                str(file_id) + '.png')

    return Yavg[-1], graph

def plot_D(parameters, graph_type, u, b, c, delta, noise, t, number_trials, num_rep, this_color, \
    rho = None, update_name = 'BD', plotting = True, show_graph = False, saving = False, color_fitness=False):    

    #matrix in which entry n,t is the concentration 
    #of the_strat at time t in trial n
    result_matrix=[]
    #run the game for each trial
    for each in range(number_trials):
        print('\n')
        print("Running trial ", each)
        #print("Evaluating trial ", each)
        graph=init.generate_weighted(parameters, graph_type)

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

        #LABEL ALLEN
        init.label_allen(graph, b, c, strat_list, start_prop_cooperators)

        #LABEL FOR A LATTICE WITH ONE SLICE OF DEFECTORS
        #init.label_BD_according_to_one_dim(graph, strat_list, parameters[1]) 
        #LABEL MULTIPLE CLIQUES 
        #init.label_dumbell_multiple_cliques(graph, strat_list, {0: 0.2, 1:0.9, 2:0, 3:0.1, 4:1})       


        this_game=game(graph, update_name, t, u, delta, plotting, show_graph, saving, color_fitness)


        if graph_type == 'triangular_lattice':
            pos = dict( (n, n) for n in graph.nodes() )
        else:
            pos = nx.spring_layout(graph)

        trial_outcome = this_game.trial(pos, num_rep, noise, graph_type, each+1)
        

        #trial_outcome=this_game.trial(graph, u, t, nx.spring_layout(graph, 1/n**.2), \
        #    graph_type, update_name, plotting, show_graph, saving, color_fitness)
        #append record for this trial of the concentrations of the_strat
        
        result_matrix.append(trial_outcome[3])

    '''
    print('~~~~~~~~~~~~~')
    print(result_matrix)
    print('~~~~~~~~~~~~~')
    '''

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

    the_avg = sum(Yavg)/len(Yavg)
    AVG=[the_avg for tictoc in range(t)]

    if plotting:
        #plot the 3 lines
        plt.plot(X, Yavg, color=this_color, marker='', linestyle = '-')
        plt.plot(X, Yplus, color=this_color, marker='', linestyle = '-')
        plt.plot(X, Yminus, color=this_color, marker='', linestyle = '-')
        #plt.plot(X, AVG, color='orange', marker='', linestyle= '-')

        #change axes ranges
        plt.xlim(0,t-1)
        plt.ylim(-.01,.01)
        #add title
        plt.title('Relationship between time and D(s) in '+str(number_trials)+ ' trials')
        #add x and y labels
        plt.ylabel('D(s)')
        plt.xlabel('Time')

        #show plot
        plt.show()
        pause(20)   
        #print("Attempting to show plot -----------------")
        #pause(60)
    #plt.close()     

    if saving:
        file_id = 0
        #randint(10**5, 10**6 - 1)
    #   print("Attempting to save plot ", data_iteration)+1
        if graph_type=='dumbell_multiple':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'size_dumbell=' + str(parameters[0]) + '_' + \
                'num_dumbell=' + str(parameters[1]) + '_' + \
                'size_path=' + str(parameters[2]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='dumbell':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'n=' + str(parameters[0]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type == 'rich_club':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'size_club=' + str(parameters[0]) + '_' + \
                'size_periphery=' + str(parameters[1]) + '_' + \
                'prob_rp=' + str(parameters[2]) + '_' + \
                'prob_rr=' + str(parameters[3]) + '_' + \
                'prob_pp=' + str(parameters[4]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='complete' or 'hypercube':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'num_nodes=' + str(parameters[0]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + \
                str(file_id) + '.png')
        elif graph_type=='triangular_lattice':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'n_dim=' + str(parameters[0]) + '_' + \
                'm_dim=' + str(parameters[1]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='random':
            plt.savefig(graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'num_nodes=' + str(parameters[0]) + '_' + \
                'ave_degree=' + str(parameters[1]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + \
                str(file_id) + '.png')

    return Yavg[-1]

def plot_D_and_coops(parameters, graph_type, u, b, c, delta, noise, t, number_trials, num_rep, \
    rho = None, update_name = 'BD', plotting = True, show_graph = False, saving = False, color_fitness=False):    

    #matrix in which entry n,t is the concentration 
    #of the_strat at time t in trial n
    result_matrix_D = []
    result_matrix_coop = []
    #run the game for each trial
    for each in range(number_trials):
        print('\n')
        print("Running trial ", each)
        #print("Evaluating trial ", each)
        graph=init.generate_weighted(parameters, graph_type)

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

        #LABEL ALLEN
        init.label_allen(graph,b,c,strat_list, start_prop_cooperators)
        #init.label_dumbell_multiple_cliques_allen(graph, b, c, strat_list, parameters[3])

        #LABEL FOR A LATTICE WITH ONE SLICE OF DEFECTORS
        #init.label_BD_according_to_one_dim(graph, strat_list, parameters[1]) 
        #LABEL MULTIPLE CLIQUES 
        #init.label_dumbell_multiple_cliques(graph, strat_list, {0: 0.2, 1:0.9, 2:0, 3:0.1, 4:1})       



        this_game=game(graph, update_name, t, u, delta, plotting, show_graph, saving, color_fitness)


        if graph_type == 'triangular_lattice':
            pos = dict( (n, n) for n in graph.nodes() )
        else:
            pos = nx.spring_layout(graph)

        trial_outcome = this_game.trial(pos, num_rep, noise, graph_type, each+1)
        

        #trial_outcome=this_game.trial(graph, u, t, nx.spring_layout(graph, 1/n**.2), \
        #    graph_type, update_name, plotting, show_graph, saving, color_fitness)
        #append record for this trial of the concentrations of the_strat
        
        result_matrix_D.append(trial_outcome[3])
        result_matrix_coop.append(trial_outcome[1]['Cooperate'])

    ########################################################
    #################### P L O T - D #######################
    ########################################################

    #---------------------X AXIS FOR BOTH-------------------
    X=[tictoc for tictoc in range(t)]
    #-------------D PLOT: average, and pm stdev-------------
    Yavg, Yplus, Yminus=[], [], []
    for tictoc in range(t):
        at_time_t=[trial[tictoc] for trial in result_matrix_D]
        #average at time t over all the trials
        average=sum(at_time_t)/len(at_time_t)
        stdev=np.std(at_time_t)
        Yavg.append(average)
        Yplus.append(average+stdev)
        Yminus.append(average-stdev)

    the_avg = sum(Yavg)/len(Yavg)
    AVG=[the_avg for tictoc in range(t)]

    if plotting:
        plt.figure(1)
        #plot the 3 lines
        plt.plot(X, Yavg, color='green', marker='', linestyle = '-')
        plt.plot(X, Yplus, color='red', marker='', linestyle = '-')
        plt.plot(X, Yminus, color='blue', marker='', linestyle = '-')
        plt.plot(X, AVG, color='orange', marker='', linestyle= '-')

        #change axes ranges
        plt.xlim(0,t-1)
        plt.ylim(-0.25,0.25)
        #add title
        plt.title('Relationship between time and D(s) in '+str(number_trials)+ ' trials')
        #add x and y labels
        plt.ylabel('D(s)')
        plt.xlabel('Time')

        #show plot
        plt.show()
        pause(5)   
        #print("Attempting to show plot -----------------")
        #pause(60)
    #plt.close()     

    if saving:
        file_id = randint(10**5, 10**6 - 1)
        #print("Attempting to save plot ", data_iteration)+1
        if graph_type=='dumbell_multiple':
            plt.savefig('DPLOT' + \
                graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'size_dumbell=' + str(parameters[0]) + '_' + \
                'num_dumbell=' + str(parameters[1]) + '_' + \
                'size_path=' + str(parameters[2]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='dumbell':
            plt.savefig('DPLOT' + \
                graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'n=' + str(parameters[0]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type == 'rich_club':
            plt.savefig('DPLOT' + \
                graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'size_club=' + str(parameters[0]) + '_' + \
                'size_periphery=' + str(parameters[1]) + '_' + \
                'prob_rp=' + str(parameters[2]) + '_' + \
                'prob_rr=' + str(parameters[3]) + '_' + \
                'prob_pp=' + str(parameters[4]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='complete' or 'hypercube':
            plt.savefig('DPLOT' + \
                graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'num_nodes=' + str(parameters[0]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + \
                str(file_id) + '.png')
        elif graph_type=='triangular_lattice':
            plt.savefig('DPLOT' + \
                graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'n_dim=' + str(parameters[0]) + '_' + \
                'm_dim=' + str(parameters[1]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='random':
            plt.savefig('DPLOT' + \
                graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'num_nodes=' + str(parameters[0]) + '_' + \
                'ave_degree=' + str(parameters[1]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + \
                str(file_id) + '.png')

    ########################################################
    ############ P L O T - C O O P - P R O P ###############
    ########################################################
    #---------------------X AXIS FOR BOTH-------------------
    X=[tictoc for tictoc in range(t)]
    #-------------D PLOT: average, and pm stdev-------------
    Yavg, Yplus, Yminus=[], [], []
    for tictoc in range(t):
        at_time_t=[trial[tictoc] for trial in result_matrix_coop]
        #average at time t over all the trials
        average=sum(at_time_t)/len(at_time_t)
        stdev=np.std(at_time_t)
        Yavg.append(average)
        Yplus.append(average+stdev)
        Yminus.append(average-stdev)

    the_avg = sum(Yavg)/len(Yavg)
    AVG=[the_avg for tictoc in range(t)]

    if plotting:
        plt.figure(2)
        #plot the 3 lines
        plt.plot(X, Yavg, color='green', marker='', linestyle = '-')
        plt.plot(X, Yplus, color='red', marker='', linestyle = '-')
        plt.plot(X, Yminus, color='blue', marker='', linestyle = '-')
        plt.plot(X, AVG, color='orange', marker='', linestyle= '-')

        #change axes ranges
        plt.xlim(0,t-1)
        plt.ylim(0,1)
        #add title
        plt.title('Relationship between time and proportion of nodes with \n strategy Cooperate in '+str(number_trials)+ ' trials')
        #add x and y labels
        plt.ylabel('Proportion of Cooperators')
        plt.xlabel('Time')

        #show plot
        plt.show()
        pause(5)   
        #print("Attempting to show plot -----------------")
        #pause(60)
    #plt.close()     

    if saving:
        file_id = 0
        print("File id is ", file_id)
        print("Attempting to save plot for prop coops")
        #print("Attempting to save plot ", data_iteration)+1
        if graph_type=='dumbell_multiple':
            plt.savefig('PROPCOOPPLOT' + \
                graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'size_dumbell=' + str(parameters[0]) + '_' + \
                'num_dumbell=' + str(parameters[1]) + '_' + \
                'size_path=' + str(parameters[2]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='dumbell':
            plt.savefig('PROPCOOPPLOT' + \
                graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'n=' + str(parameters[0]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type == 'rich_club':
            plt.savefig('PROPCOOPPLOT' + \
                graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'size_club=' + str(parameters[0]) + '_' + \
                'size_periphery=' + str(parameters[1]) + '_' + \
                'prob_rp=' + str(parameters[2]) + '_' + \
                'prob_rr=' + str(parameters[3]) + '_' + \
                'prob_pp=' + str(parameters[4]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='complete' or 'hypercube':
            plt.savefig('PROPCOOPPLOT' + \
                graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'num_nodes=' + str(parameters[0]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + \
                str(file_id) + '.png')
        elif graph_type=='triangular_lattice':
            plt.savefig('PROPCOOPPLOT' + \
                graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'n_dim=' + str(parameters[0]) + '_' + \
                'm_dim=' + str(parameters[1]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + str(file_id) + '.png')
        elif graph_type=='random':
            plt.savefig('PROPCOOPPLOT' + \
                graph_type + '_' + \
                update_name + '_' + \
                'u=' + str(u) + '_' + \
                'noise=' + str(noise) + '_' + \
                'num_nodes=' + str(parameters[0]) + '_' + \
                'ave_degree=' + str(parameters[1]) + '_' + \
                'prop_coop=' + str(start_prop_cooperators) + '_' + \
                str(number_trials) + 'trials' + '_' + \
                str(t) + 'timesteps' + '_' + \
                'b_over_c=' + str(b) + '_' + \
                str(file_id) + '.png')


    return Yavg[-1]

'''---------------------------------------------------
    CALCULATIONS FROM THE PAPER 'ANY POP STRUCTURE'
----------------------------------------------------'''
'''
def prob_n_step_walk(graph, i, j, n):
    w_i = 0
    for neighbor in graph.neighbors(i):
        w_i += graph[i][neighbor]['weight']
    p_ij_sum = 0
    if n == 2:
        print("We've gotten to the n=2")
        #want to determine if there is a path from i to k to j
        for k in graph.neighbors(i):
            #there is a path from i to k
            if j in graph.neighbors(k):
                #there is a path from k to j
                p_ik = graph[i][k]['weight']/w_i
                p_kj = graph[k][j]['weight']/w_i
                p_ij_sum += p_ik * p_kj
    else:
        p_ij_sum = graph[i][j]['weight'] / w_i 
    return p_ij_sum
def reproductive_value(graph, i):
    w_i = 0
    for neighbor in graph.neighbors(i):
        w_i += graph[i][neighbor]['weight']
    W_sum = 0
    for j in graph.neighbors(i):
        W_sum += graph[i][j]['weight']
    return w_i/W_sum
def s_indicator(graph, i):
    if graph.node[i]['strategy'] == 'Cooperate':
        s_i = 1
    else:
        s_i = 0
    return s_i
def edge_weighted_payoff(graph, i, n):
    s_i = graph.node[i]['s']
    expectation = 0
    for j in graph.neighbors(i):
        print("Node i is ", i)
        print("Node j is ", j)
        expectation += prob_n_step_walk(graph, i, j, n) * s_indicator(graph, j)
    f = -c * s_i + b * expectation
    return f
'''

def D(graph, delta, b, c):
    d_sum = 0
    for i in graph.nodes():
        f_0i = graph.node[i]['f0']
        '''
        print("f_0i code worked")
        '''
        f_2i = graph.node[i]['f2']
        d_sum += graph.node[i]['pi'] * graph.node[i]['s'] * (f_0i - f_2i)
    return delta * d_sum



'''--------------------------------------------
                        TESTS
--------------------------------------------'''

'''--------------------------
        IMPORTANT VARS
--------------------------'''

'''--------------VARIABLES ALWAYS USED ----------------'''
strat_list  = ['Cooperate', 'Defect']
graph_type  = 'dumbell_multiple'
update_name = 'BD'


u       = 0.0001
delta   = 0.005
noise   = 0
b       = 2
max_b   = 2
c       = 1
t       = 300

start_prop_cooperators  = 0.7
number_trials           = 1
#Number of nodes to reproduce at each timestep 
num_rep = 5


#To run a particular graph simulation, uncomment the parameters line


'''-----------Lattice Variables----------------'''
#n_lattice = 20
#m_lattice = 50
#list of parameters that will be used to build graph
#parameters = [n_lattice,m_lattice]


'''-----------Multiple Dumbell Variables----------------'''
#size_dumbell= 15
#num_dumbell = 2
#size_path   = 4
#list of parameters that will be used to build graph
#parameters = [size_dumbell, num_dumbell, size_path]


'''-----------Multiple Dumbell, Multiple Proportions Variables----------------'''
size_dumbell= 60
num_dumbell = 2
size_path   = 10


#cliques_to_proportions = {0 : 1, 1 : 1, 2:1, 3:1, 4:1}
cliques_to_proportions = {0 : 1, 1 : 1, 2:0.5, 3:0.5}

# 2:0, 3:0.1, 4:1, 5:.5}
#6:.4, 7:.5, 8:.2, 9:.6}
#list of parameters that will be used to build graph
#parameters = [size_dumbell, num_dumbell, size_path, cliques_to_proportions]


'''-----------Complete Bipartite Graph Variables----------------'''

#side_1 = 30
#side_2 = 30
#parameters = [side_1, side_2]

'''-----------Rich Club Variables----------------'''
#size_club       = 10
#size_periphery  = 20
#prob_rp         = .2
#prob_rr         = .2
#prob_pp         = .2
#list of parameters that will be used to build graph
#parameters = [size_club, size_periphery, prob_rp, prob_rr, prob_pp]

'''-----------String Dumbell Variables----------------'''
#size of each dumbell in the string
#sizes=[15,15]
#lengths of the uniting paths
#lengths=[3]
#cliques_to_proportions = {0 : 1, 1 : 1, 2:1, 3:1, 4:1}

#parameters=[sizes,lengths, cliques_to_proportions]





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

'''------------------------
CODE FOR LATTICES
-------------------------'''
#for start_prop_cooperators in prop_increments:
#plot_lattice_density_and_payoff(parameters, graph_type, u, t, max_b, 'Cooperate', update_name = 'BD', \
#    plotting = True, show_graph = False, saving = True, color_fitness = True)


'''------------------------
CODE FOR MULTIPLE DUMBELLS
-------------------------'''
'''
prop_increments = np.arange(.1, 1.1, 0.1)
for start_prop_cooperators in prop_increments:
    start_prop_cooperators = round(start_prop_cooperators, 3)
    plt.gcf().clear()
    plot_many_trials(parameters, graph_type, u, delta, noise, t, number_trials, 'Cooperate', num_rep, None, 'BD', \
    plotting=True, show_graph=False, saving=True, color_fitness=True)
'''

'''------------------------
CODE FOR MULTIPLE DUMBELLS
AND MULTIPLE PROPORTIONS
-------------------------'''

#plot_multiple_dumbell_each_clique(parameters, graph_type, u, b, c, delta, noise, t, number_trials, 'Cooperate', num_rep, None, \
#    'BD', plotting=True, show_graph=True, saving=False, color_fitness=True)


#plot_trial_until_stable(parameters, graph_type, u, t, 'Cooperate', num_rep, \
#    update_name = 'BD', plotting = True, show_graph = True, saving = False, color_fitness=False)

'''
graph=init.generate_graph([[5,7,9],2], 'dumbell_multiple_sized')
init.label_dumbell_multiple_cliques_precise(graph, strat_list, {0:0.2, 1:0.9, 2:0.9})
dis.color_and_draw_graph(graph)
dis.color_fitness_and_draw_graph(graph, nx.spring_layout(graph))
print(get_props_cliques(graph))
'''

#graph = plot_many_trials(parameters, graph_type, u, delta, noise, t, number_trials, 'Cooperate', num_rep, \
#    rho = None, update_name = 'BD', plotting = True, show_graph = False, saving = False, color_fitness=True)[1]

'''------------------------
UTKOVSKI TRIALS
-------------------------'''
#this_lambda=0.5
#kappa=0.5






num_dumbell = 2
size_path   = 4

#cliques_to_proportions = {0 : 1, 1 : 1, 2:1, 3:1, 4:1}
cliques_to_proportions = {0 : .5, 1 : .5}
# 2:0, 3:0.1, 4:1, 5:.5}
#6:.4, 7:.5, 8:.2, 9:.6}
#list of parameters that will be used to build graph
num_rep = 5

number_trials = 1

size_dumbell = 30
#for size_dumbell in range(2, 50):
parameters = [size_dumbell, num_dumbell, size_path, cliques_to_proportions]



remaining_colors = list(mcolors.CSS4_COLORS.keys())
for color in light_colors:
    remaining_colors.remove(color) 

'''
for i in range(10):
    this_color = color_list.pop()
    if i == 10:
        save_boolean = True
    else:
        save_boolean = False
    plot_D(parameters, graph_type, u, b, c, delta, noise, t, number_trials, num_rep, this_color, \
        rho = None, update_name = 'BD', plotting = True, show_graph = False, saving = save_boolean, color_fitness=True)

#plt.gcf().clear()
'''

######## T E S T - I S L A N D - M O D E L #########
graph_type='with_indicator'

'''------------
    TEST 1
-------------'''
#MAKING OF INDICATOR GRAPH
indicator=nx.Graph()
sizes={0:40,
        1:40,
        2:40,
        3:40}

strengths={(0,1):0.2,
        (0,3):0.01,
        (1,2):0.01,
        (2,3):0.01}
#ADD EDGES
indicator.add_edges_from(strengths.keys())
#SET ATTRIBUTES
nx.set_node_attributes(indicator, name='size', values=sizes)
nx.set_edge_attributes(indicator, name='strength', values=strengths)
#PARAMETERS
cliques_to_proportions = {0 : 0.9, 1 : 0.1, 2:0.8, 3:0.8}
parameters=[indicator, None, cliques_to_proportions]

'''------------
    TEST 1
-------------'''
#MAKING OF INDICATOR GRAPH
indicator=nx.Graph()
sizes={0:40,
        1:40}

strengths={(0,1):0.0002}
#ADD EDGES
indicator.add_edges_from(strengths.keys())
#SET ATTRIBUTES
nx.set_node_attributes(indicator, name='size', values=sizes)
nx.set_edge_attributes(indicator, name='strength', values=strengths)
#PARAMETERS
cliques_to_proportions = {0 : 0.9, 1 : 0.1}
parameters=[indicator, None, cliques_to_proportions]



iterations=3
for i in range(iterations):
    this_color = remaining_colors.pop()
    if i == iterations-1:
        save_boolean = True
    else:
        save_boolean = False
    plot_multiple_dumbell_each_clique(parameters, graph_type, u, b, c, delta, noise, t, number_trials, 'Cooperate', num_rep, this_color, \
        rho = None, update_name = 'BD', plotting = True, show_graph = False, saving = save_boolean, color_fitness=False)




#graph = plot_many_trials_utkovski(parameters, graph_type, strat_list, start_prop_cooperators, u, this_lambda, kappa, noise, t, number_trials, \
#    rho=None, plotting = True, show_graph = True, saving = False, color_fitness=True)[1]

#print(D(graph, delta, b, c))
#plot_D_and_coops(parameters, graph_type, u, b, c, delta, noise, t, number_trials, num_rep, \
#    rho = None, update_name = 'DB', plotting = True, show_graph = False, saving = True, color_fitness=False)

