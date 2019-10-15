# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:04:52 2019

@author: vaio
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import Analysis_Tools_Homans
import networkx as nx
import os

N,T,memory_size = [100,1000,10]

current_path = os.getcwd()
version = '\\master'
path = '\\runned_files'+version+'\\N%d_T%d_memory_size%d\\'%(N,T,memory_size)

with open(current_path+path+'Tracker.pkl','rb') as tracker_file:
    tracker = pickle.load(tracker_file)

with open(current_path+path+'Agents.pkl','rb') as agent_file:
    a_matrix = pickle.load(agent_file)

analyse = Analysis_Tools_Homans.Analysis(N,T,memory_size,a_matrix)

#analyse.draw_graph_weighted_colored()
#analyse.draw_graph()

graph_type = 'trans_number'
constructed_graph = analyse._graph_construction(graph_type)
nx.write_gexf(constructed_graph,current_path+path+'%s_graph.gexf'%(graph_type))

dynamic_graph = tracker.make_dynamic_trans_time_graph(constructed_graph)
nx.write_gexf(dynamic_graph,current_path+path+'dynamic_%s_graph.gexf'%(graph_type))

a_money       = analyse.array('money')
#a_approval    = analyse.array('approval')
#a_worth_ratio = analyse.array('worth_ratio')
#a_neighbour   = analyse.array('neighbour')
a_value       = analyse.array('value')
#a_time        = analyse.array('time')
a_probability = analyse.array('probability')
#a_utility = analyse.array('utility')

analyse.hist('money')
analyse.hist_log_log('money')
analyse.hist('approval')
analyse.hist_log_log('approval')
analyse.hist('degree')
analyse.hist_log_log('degree')
analyse.hist('value')
analyse.hist_log_log('value')
analyse.hist('probability')
analyse.hist_log_log('probability')
analyse.hist('utility')
analyse.hist_log_log('utility')


analyse.topology_chars()

analyse.agents_prob_sum()

a_money       = analyse.array('money')
#a_approval    = analyse.array('approval')
#a_worth_ratio = analyse.array('worth_ratio')
#a_neighbour   = analyse.array('neighbour')
a_value       = analyse.array('value')
#a_time        = analyse.array('time')
a_probability = analyse.array('probability')
#a_utility = analyse.array('utility')
#tracker.plot('self_value',title='Self Value')
#tracker.plot('valuable_to_others',title='How Much Valuable to Others')
#tracker.plot('worth_ratio',title='worth_ratio Evolution by Time')
tracker.trans_time_visualizer(3,'Transaction Time Tracker')

#tracker.plot_general(num_transaction_tot, title='Number of Transaction Vs. Time')
#tracker.plot_general(num_explore, title='Number of Explorations Vs. Time')
##tracker.plot_general(agreement_tracker, title='agreement Point')
##tracker.plot_general(acceptance_tracker, title='Threshold Acceptance Over Time')
#
#plt.figure()
#plt.plot(p0_tracker[::2])
#plt.plot(p1_tracker[::2])
#plt.plot(p2_tracker[::2])
#plt.title('P0 & P1 & P2')
#tracker.hist_general(p0_tracker,title='p0')
#tracker.hist_general(p1_tracker,title='p1')
#tracker.hist_general(p2_tracker,title='p2')
#tracker.hist_log_log_general(p0_tracker,title='P0')
#tracker.hist_log_log_general(p1_tracker,title='P1')
#tracker.hist_log_log_general(p2_tracker,title='P2')

#plt.figure()
#for i in np.arange(N):
#    plt.plot(a_matrix[i].neighbour)
#plt.title('A[i]s number of transaction with others')

analyse.degree_vs_attr()

#plt.figure()
#for i in np.arange(N):
#    plt.plot(similarity_tracker[i])
#plt.ylim([0,1.1])
#plt.title('Similarity Tracker')

#print(analyse.segregation())

tracker.hist_general(a_probability[a_probability>0.1])
tracker.hist_log_log_general(a_probability[a_probability>0.1])
array = a_value[a_value>0.8]
plt.figure()
plt.xscale('log')
plt.yscale('log')
bins=np.logspace(np.log10(np.amin(array)),np.log10(np.amax(array)),12)
plt.hist(array,bins=bins)
plt.title('Probability log-log Historgram bigger than 0.1')


