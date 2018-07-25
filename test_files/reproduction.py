'''-------------------
        PACKAGES
-------------------'''

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

'''-------------------
    IMPORTED FILES
-------------------'''
import graph_initialization as init

'''------
REPRODUCTION
make all of the reproduction process based on the fitness, weight and mutation parameters.
--------'''

def birth_death(G, strat_list, u, num_rep):
    '''
    print('REPRODUCTION')
    print('---------')
    print('-> number of reproducing nodes: '+str(num_rep))
    '''

    #CHOOSE REPRODUCING NODES
    fitness_dict = nx.get_node_attributes(G, 'fitness')
    #print("The fitness dictionary is ", fitness_dict)
    fitness_sum = sum(fitness_dict.values())
    #print("The fitness sum is ", fitness_sum)
    cutoffs=[]
    for num in range(num_rep):
        cutoffs.append(random.uniform(0, fitness_sum))
    #print("The cutoffs are "+ str(cutoffs))
    reproduced_nodes=set()
    for cut in cutoffs:
        last=0
        marker=0
        for n in nx.nodes(G):
            #print(n)
            last=marker
            marker+=nx.get_node_attributes(G, 'fitness')[n]
            if last<cut<marker:
                if list(G.adj[n].keys()) != []:
                    reproduced_nodes.add(n)
                break
    old_strategies=[]
    reproduced_strategies=[]
    inheritance = {}
    '''
    print('-> Reproduced nodes: ')
    print('')
    print(reproduced_nodes)
    '''
    for i in reproduced_nodes:
        reproduced_strategy=G.node[i]['strategy']
        #set reproduced strategy
        j = random.choice(list(G.adj[i].keys()))
        old_strategy = G.node[j]['strategy']

        difference=G.node[i]['fitness']-G.node[j]['fitness']
        if random.random()<=q(difference, .1, 6):
            '''
            print(str(i) + ' replaced '+ str(j))
            '''
            #now decide which strategy will be inherited!
            #i.e. is there a mutation?
            mistake_indicator = random.uniform(0, 1)
            if mistake_indicator < u:
                #print("There has been a mutation!")
                mutation_list = [x for x in strat_list if x != reproduced_strategy]
                
                if mutation_list == []:
                    #print("There cannot be mutations in this population.")
                    inheritance[j] = reproduced_strategy
                else:
                    inheritance[j] = np.random.choice(mutation_list)
                    reproduced_strategy = inheritance[j]
                #print("Node ", j, " now has strategy ", G.node[j]['strategy'])        
            else:
                # there is not a mutation
                inheritance[j] = reproduced_strategy
            # Node j has now just been born, so we set its fitness to 0
            G.node[j]['fitness'] = random.uniform(0,1)
            reproduced_strategies.append(reproduced_strategy)
            old_strategies.append(old_strategy)
    #do all the replacements according to inheritance
    '''
    print('inheritance dictionary: ')
    print()
    print(inheritance)
    '''
    for j in inheritance.keys():
        G.node[j]['strategy'] = inheritance[j]
        if inheritance[j] == 'Cooperate':
            G.node[j]['s'] = 1
        else:
            G.node[j]['s'] = 0

    return [G, reproduced_strategies, old_strategies, reproduced_nodes]

def death_birth(G, strat_list, u, num_rep):
    #CHOOSE num_rep REPLACED NODES 
    replaced_nodes = random.sample(list(G.nodes()),num_rep)
    old_strategies = []
    for replaced in replaced_nodes:
        old_strategies.append(G.node[replaced]['strategy'])  
    
    #get fitnesses
    fits=nx.get_node_attributes(G, 'fitness')
    #initialize nodes which are going to reproduced  
    #and their respective strategies
    reproduced_nodes = []
    reproduced_strategies = []

    for replaced in replaced_nodes:
        #dictionary for neighbors to compete for replacement
        competition={}
        for n in list(G.neighbors(replaced)):
            m=min([n,replaced])
            M=max([n,replaced])
            competition[n]=fits[n]* weights[(m,M)]

        #pool, take cutoffs, decide which 
        sum_competition = sum(competition.values())
        cutoff = random.uniform(0,sum_competition)
        marker = 0
        prev = 0
        print(G.neighbors(replaced))
        for n in G.neighbors(replaced):
            prev = marker
            marker+=competition[n]
            if prev<cutoff<marker:
                if list(G.adj[n].keys()) != []:
                    reproduced_nodes.append(n)
                    if random.random()<u:
                        #there was a mutation
                        reproduced_strategies.append(random.choice(strat_list))
                    else:
                        reproduced_strategies.append(G.node[n]['strategy'])
                break
        for index in len(replaced_nodes):
            i = replaced_nodes[index]
            G.node[i]['strategy'] = reproduced_strategies[index]
            if G.node[i]['strategy'] == 'Cooperate':
                G.node[i]['s'] = 1
            else:
                G.node[i]['s'] = 0

    return [G, reproduced_strategies, old_strategies, reproduced_nodes]

def pairwise_comparison(G, strat_list, u, num_rep):
    #Sample for reproductors
    replaced_nodes=random.sample(nx.nodes(G), num_rep)
    #replaced=random.choice(nx.nodes(G))
    old_strategy=G.node[replaced]['strategy']
    #competition with e_ij between neighbors
    weights=nx.get_edge_attributes(G, 'weight')
    competition={}
    for n in G.neighbors(replaced):
        competition[n]=weights[(n,replaced)]
    sum_competition=sum(competition.values())
    cutoff=random.uniform(0,sum_competition)
    sum_so_far=0
    for node in G.neighbors(replaced):
        sum_so_far+=competition[node]
        if cutoff<sum_so_far:
            reproduced=node
    #replacement with probability Theta(F_i-F_j)
    difference=G.node[reproduced]['fitness']-G.node[replaced]['fitness']
    if random.random()<=q(difference, .1, 3):
    #if random.random()<=Theta(difference):
        #place reproduced into replaced,consider mutation
        mistake_indicator = random.uniform(0, 1)
        if mistake_indicator<u:
            #there is a mutation
            G.node[replaced]['strategy']=random.choice(strat_list)
        else:
            G.node[replaced]['strategy']=G.node[reproduced]['strategy']
    return [G, G.node[replaced]['strategy'], old_strategy]

def imitation(G, strat_list, u):
    #Choose node uniformly to be replaced
    replaced=random.choice(nx.nodes(G))
    old_strategy=G.node[replaced]['strategy']
    #competition with e_ij between neighbors
    weights=nx.get_edge_attributes(G, 'weight')
    competition={}
    for n in G.neighbors(replaced):
        competition[n]=G.node[n]['fitness']
    sum_competition=sum(competition.values())
    cutoff=random.uniform(0,sum_competition)
    sum_so_far=0
    for node in G.neighbors(replaced):
        sum_so_far+=competition[node]
        if cutoff<sum_so_far:
            reproduced=node
    #place reproduced into replaced,consider mutation
    mistake_indicator = random.uniform(0, 1)
    if mistake_indicator<u:
        #there is a mutation
        G.node[replaced]['strategy']=random.choice(strat_list)
    else:
        G.node[replaced]['strategy']=G.node[reproduced]['strategy']
    return [G, G.node[replaced]['strategy'], old_strategy]

'''--------------
 HELPER FUNCTIONS
--------------'''

def Theta(val):
    V=1+math.exp(-val)
    return 1/V

def q(val, K, d):
    val=round(val, 6)
    V = 1+ math.exp(val/(K*d))
    return 1/V 

