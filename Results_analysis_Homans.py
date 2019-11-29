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

N = 100
T = N*20
memory_size = 10
current_path = os.getcwd()
version = 'community_isomorphism'
path = '\\runned_files'+'\\N%d_T%d\\'%(N,T)+version+'\\'
path = current_path+path

with open(path+'Other_data.pkl','rb') as data_file:
    num_transaction_tot = pickle.load(data_file)
#    trans_time          = pickle.load(data_file)
    explore_prob_arr    = pickle.load(data_file)
    
with open(path+'Agents.pkl','rb') as agent_file:
    a_matrix = pickle.load(agent_file)

with open(path + 'Tracker.pkl','rb') as tracker_file:
    tracker = pickle.load(tracker_file)
    
""" Analysis Plots""" 
analyse = Analysis_Tools_Homans.Analysis(N,T,memory_size,a_matrix,path)
analyse.graph_construction('trans_number',num_transaction_tot,trans_time=trans_time)
analyse.draw_graph_weighted_colored()
analyse.graph_correlations()


#
#constructed_graph = analyse.G
#dynamic_graph = tracker.make_dynamic_trans_time_graph(constructed_graph)
#nx.write_gexf(dynamic_graph,path+'dynamic_trans_number_graph.gexf')

"""tracker plots"""
tracker.get_path(path)
tracker.valuability()

#for prop in ['money','asset','approval']:
#    tracker.property_evolution(prop)
#    tracker.correlation_growth_situation(prop,'situation')

#tracker.correlation_growth_situation('money','initial_money')
#tracker.correlation_growth_situation('asset','initial_asset')
#tracker.correlation_growth_situation('approval','initial_approval')

#tracker.plot('worth_ratio',title='Worth_ratio Evolution by Time',alpha=1)
#tracker.plot('correlation_mon',title='Correlation of Money and Situation')
#tracker.plot('correlation_situ',title="Correlation of Situation and Neighbor's Situation")

""" community isomorphism investigation"""

analyse.community_detection()

for prop in ['money','asset','approval','worth_ratio']:
    analyse.communities_property_hist(prop)
    




