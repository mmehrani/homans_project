# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:04:52 2019

@author: Taha Enayat, Mohsen Mehrani
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import Analysis_Tools_Homans
import networkx as nx
import os

N,T,memory_size = [100,1000,10]

current_path = os.getcwd()
version = '\\12_threshold_percentage_4_10'
path = '\\runned_files'+version+'\\N%d_T%d\\'%(N,T)

with open(current_path+path+'Tracker.pkl','rb') as tracker_file:
    tracker = pickle.load(tracker_file)

with open(current_path+path+'Agents.pkl','rb') as agent_file:
    a_matrix = pickle.load(agent_file)

num_transaction_tot = np.load(path+'num_transaction_tot.npy')
explore_prob_array = np.load(path+'explore_prob_array.npy')


""" Analysis Plots""" 
analyse = Analysis_Tools_Homans.Analysis(N,T,memory_size,a_matrix,path)
analyse.graph_construction('trans_number',num_transaction_tot,explore_prob_array,tracker_obj=tracker)
analyse.draw_graph_weighted_colored()
analyse.graph_correlations()

graph_type = 'trans_number'
mid_constructed_graph = analyse._graph_construction(graph_type,tracker_obj = tracker,sampling_time = int(T/2))
nx.write_gexf(mid_constructed_graph,current_path+path+'%s_graph.gexf'%(graph_type))

constructed_graph = analyse._graph_construction(graph_type)
dynamic_graph = tracker.make_dynamic_trans_time_graph(constructed_graph)
nx.write_gexf(dynamic_graph,current_path+path+'dynamic_%s_graph.gexf'%(graph_type))


"""tracker plots"""
tracker.get_path(path)

for agent in analyse.rich_agents_in_communities():
    tracker.trans_time_visualizer(agent,'Transaction Time Tracker money:%f'%(a_matrix[agent].money))

tracker.valuability()

for prop in ['money','asset','approval']:
    tracker.property_evolution(prop)
    tracker.correlation_growth_situation(prop,'situation')

tracker.correlation_growth_situation('money','initial_money')
tracker.correlation_growth_situation('asset','initial_asset')
tracker.correlation_growth_situation('approval','initial_approval')

tracker.plot('worth_ratio',title='Worth_ratio Evolution by Time',alpha=1)
tracker.plot('correlation_mon',title='Correlation of Money and Situation')
tracker.plot('correlation_situ',title="Correlation of Situation and Neighbor's Situation")






