import numpy as np
import pandas as pd
import holoviews as hv
import matplotlib.pyplot as plt
import networkx as nx

#from collections import OrderedDict

import graph_initialization as init
'''
strat_list = ['Cooperate', 'Defect']
start_prop_cooperators = .5

G = init.generate_dumbell_multiple_cliques(10,5,1)
init.label_birth_death(G, strat_list, start_prop_cooperators)
'''
#color_palette = sns.cubehelix_palette(3)
#cmap = {k:color_palette[v-1] for k,v in zip(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],[2, 1, 3, 2])}


'''

cmaps = OrderedDict()

viridis = cmaps['Perceptually Uniform Sequential'][0]

nx.draw(G, node_color=[ viridis[G.node[n]['fitness'] ] for n in G.nodes()], node_shape='^', with_labels=True, hold=True)
'''


'''
val_map = {}
for n in nx.nodes(G):
    val_map[n] = G.node[n]['fitness']
    print("Fitness of node ", n, " is ", G.node[n]['fitness'])

values = [val_map.get(node, 0.25) for node in G.nodes()]

#pos = nx.spring_layout(G, iterations=200)
nx.draw(G, color_map=values, node_shape='^', with_labels=True, hold=True, cmap=plt.get_cmap('jet'))
plt.show()
'''

'''

hv.extension('bokeh')

N = 8
node_indices = np.arange(N, dtype=np.int32)
source = np.zeros(N, dtype=np.int32)
target = node_indices

padding = dict(x=(-1.2, 1.2), y=(-1.2, 1.2))

simple_graph = hv.Graph(((source, target),)).redim.range(**padding)
simple_graph





def bezier(start, end, control, steps=np.linspace(0, 1, 100)):
    return (1-steps)**2*start + 2*(1-steps)*steps*control+steps**2*end

x, y = simple_graph.nodes.array([0, 1]).T

paths = []
for node_index in node_indices:
    ex, ey = x[node_index], y[node_index]
    paths.append(np.column_stack([bezier(x[0], ex, 0), bezier(y[0], ey, 0)]))
    
bezier_graph = hv.Graph(((source, target), (x, y, node_indices), paths)).redim.range(**padding)
bezier_graph
'''


def color_and_draw_graph(G):
    '''
    INPUTS:     Graph with strategy node attributes
    OUTPUTS:    Graph where color maps has colors according to 
                node strategy attributes
    '''
    # initializes color map
    color_map = []
    for n in nx.nodes(G):
        if G.node[n]['strategy'] == 'Cooperate':
            color_map.append('green')
        else:
            color_map.append('red')

    # draws colored graph
    #plt.ion()
    nx.draw(G,node_color = color_map,with_labels = True)
    plt.show()
    #plt.pause(2.0)

    return G

def color_fitness_and_draw_graph(G):
    '''
    INPUTS:     Graph with strategy node attributes
    OUTPUTS:    Graph where color maps has colors according to 
                node strategy attributes
    '''
    # initializes color map
    node_labels = nx.get_node_attributes(G,'fitness')

    coop_labels = {}
    defect_labels = {}

    for i in node_labels:
        node_labels[i] = round(node_labels[i], 4)
        if G.node[i]['strategy'] == 'Cooperate':
            coop_labels[i] = node_labels[i]
        else:
            defect_labels[i] = node_labels[i]

    pos = nx.spring_layout(G, iterations=200)

    #Display cooperator nodes
    nx.draw_networkx_nodes(G, pos, nodelist=coop_labels.keys(), node_color=[fitness for fitness in coop_labels.values()], node_shape='o', cmap=plt.cm.YlOrRd)

    #Display defector nodes
    nx.draw_networkx_nodes(G, pos, nodelist=defect_labels.keys(), node_color=[fitness for fitness in defect_labels.values()], node_shape='^', cmap=plt.cm.YlOrRd)

    #Display edges
    nx.draw_networkx_edges(G, pos)

    #Display node labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, ax=None)

    plt.show()




'''
    val_map = {}
    for n in nx.nodes(G):
        val_map[n] = G.node[n]['fitness']
        #if G.node[n]['strategy'] == 'Cooperate':
        #    shapes[n] = 'o'
        #else:
        #    shapes[n] = '^'
    values = [val_map.get(node, 0.25) for node in G.nodes()]

    # draws colored graph
    #plt.ion()
    nx.draw(G, cmap=plt.get_cmap('jet'), node_color = values, node_shape = '^', with_labels = True, hold=True)
    plt.show()
    #plt.pause(2.0)

    return G
'''