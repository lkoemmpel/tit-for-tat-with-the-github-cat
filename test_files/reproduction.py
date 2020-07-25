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

def birth_death(G, strat_list, u, num_rep, b, c):
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
        reproduced_strategy=G.nodes[i]['strategy']
        #set reproduced strategy
        j = random.choice(list(G.adj[i].keys()))
        old_strategy = G.nodes[j]['strategy']

        difference=G.nodes[i]['fitness']-G.nodes[j]['fitness']
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
                #print("Node ", j, " now has strategy ", G.nodes[j]['strategy'])        
            else:
                # there is not a mutation
                inheritance[j] = reproduced_strategy
            # Node j has now just been born, so we set its fitness to 0
            G.nodes[j]['fitness'] = random.uniform(0,1)
            reproduced_strategies.append(reproduced_strategy)
            old_strategies.append(old_strategy)
    #do all the replacements according to inheritance
    '''
    print('inheritance dictionary: ')
    print()
    print(inheritance)
    '''
    for j in inheritance.keys():
        G.nodes[j]['strategy'] = inheritance[j]
        if inheritance[j] == 'Cooperate':
            G.nodes[j]['s'] = 1
        else:
            G.nodes[j]['s'] = 0

    for i in G.nodes():
        G.nodes[i]['f0'] = -c*G.nodes[i]['s']
        for j in G.neighbors(i):
            G.nodes[i]['f0'] += b*init.prob_n_step_walk(G,i,j,1)*G.nodes[j]['s']

        G.nodes[i]['f2'] = -c*G.nodes[i]['s']
        for j in G.neighbors(i):
            G.nodes[i]['f2'] += b*init.prob_n_step_walk(G,i,j,2)*G.nodes[j]['s']

    return [G, reproduced_strategies, old_strategies, reproduced_nodes]

def death_birth(G, strat_list, u, num_rep, b, c):
    #CHOOSE num_rep REPLACED NODES 
    replaced_nodes = random.sample(list(G.nodes()),num_rep)
    old_strategies = []
    for replaced in replaced_nodes:
        old_strategies.append(G.nodes[replaced]['strategy'])  
    
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
                        reproduced_strategies.append(G.nodes[n]['strategy'])
                break
        for index in len(replaced_nodes):
            i = replaced_nodes[index]
            G.nodes[i]['strategy'] = reproduced_strategies[index]
            if G.nodes[i]['strategy'] == 'Cooperate':
                G.nodes[i]['s'] = 1
            else:
                G.nodes[i]['s'] = 0

    for i in G.nodes():
        G.nodes[i]['f0'] = -c*G.nodes[i]['s']
        for j in G.neighbors(i):
            G.nodes[i]['f0'] += b*init.prob_n_step_walk(G,i,j,1)*G.nodes[j]['s']

        G.nodes[i]['f2'] = -c*G.nodes[i]['s']
        for j in G.neighbors(i):
            G.nodes[i]['f2'] += b*init.prob_n_step_walk(G,i,j,2)*G.nodes[j]['s']  

    return [G, reproduced_strategies, old_strategies, reproduced_nodes]

def pairwise_comparison(G, strat_list, u, num_rep):
    #Sample for reproductors
    replaced_nodes=random.sample(nx.nodes(G), num_rep)
    #replaced=random.choice(nx.nodes(G))
    old_strategies=[G.nodes[replaced]['strategy'] for replaced in replaced_nodes]
    #old_strategy=G.nodes[replaced]['strategy']
    reproduced_nodes=[]

    #competition with e_ij between neighbors
    weights=nx.get_edge_attributes(G, 'weight')
    competition={}
    for n in G.neighbors(replaced):
        m=min(n,replaced)
        M=max(n,replaced)
        competition[n]=weights[(m,M)]
    sum_competition=sum(competition.values())
    cutoff=random.uniform(0,sum_competition)
    prev=0
    mark=0
    for node in G.neighbors(replaced):
        prev=mark
        mark+=competition[node]
        if prev<cutoff<mark:
            reproduced_nodes.append(node)
    inheritance = {}
    for index in len(reproduced_nodes):
        replaced = replaced_nodes[index]
        reproduced = reproduced_nodes[index]
        #replacement with probability Theta(F_i-F_j)
        difference = G.nodes[reproduced]['fitness']-G.nodes[replaced]['fitness']
        if random.random() <= q(difference, .1, 3):
            #place reproduced into replaced,consider mutation
            if random.uniform(0, 1) < u:
                #there is a mutation
                inheritance[replaced] = random.choice(strat_list)
            else:
                inheritance[replaced] = G.nodes[reproduced]['strategy']

    
    for replaced in inheritance.keys():
        G.nodes[replaced]['strategy']=inheritance[replaced]


    return [G, list(inheritance.values()), old_strategies]

def imitation(G, strat_list, u):
    #Choose node uniformly to be replaced
    replaced=random.choice(nx.nodes(G))
    old_strategy=G.nodes[replaced]['strategy']
    #competition with e_ij between neighbors
    weights=nx.get_edge_attributes(G, 'weight')
    competition={}
    for n in G.neighbors(replaced):
        competition[n]=G.nodes[n]['fitness']
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
        G.nodes[replaced]['strategy']=random.choice(strat_list)
    else:
        G.nodes[replaced]['strategy']=G.nodes[reproduced]['strategy']
    return [G, G.nodes[replaced]['strategy'], old_strategy]

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

