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
import matplotlib.animation as animation
import graph_tools_glossary

N = 100
T = N*10 + 500
memory_size = 10
version = 'friendship_point'
current_path = os.getcwd()
path = current_path + '\\runned_files'+'\\N%d_T%d\\'%(N,T)+version+'\\'

"""Open File"""
with open(path+'Other_data.pkl','rb') as data_file:
    num_transaction_tot = pickle.load(data_file)
    explore_prob_arr    = pickle.load(data_file)
    
with open(path+'Agents.pkl','rb') as agent_file:
    a_matrix = pickle.load(agent_file)

with open(path + 'Tracker.pkl','rb') as tracker_file:
    tracker = pickle.load(tracker_file)
""""""
tool = graph_tools_glossary.Graph_related_tools(1200,N,a_matrix)
tool.graph_construction('trans_number',num_transaction_tot,sample_time_trans = tracker.sample_time_trans)
""" Plots""" 
analyse = Analysis_Tools_Homans.Analysis(N,T,memory_size,a_matrix,path)
analyse.graph_construction('trans_number',num_transaction_tot,sample_time_trans = tracker.sample_time_trans)
analyse.draw_graph_weighted_colored()
analyse.graph_correlations()

#constructed_graph = analyse.G
#dynamic_graph = tracker.make_dynamic_trans_time_graph(constructed_graph)
#nx.write_gexf(dynamic_graph,path+'dynamic_trans_number_graph.gexf')

tracker.get_path(path)
tracker.valuability()
tracker.plot_general(explore_prob_arr,title='Average Exploration Probability')

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
analyse.hist('asset')
analyse.hist_log_log('asset')

analyse.money_vs_situation(path+'money_vs_situation')
analyse.transaction_vs_asset()
#analyse.degree_vs_attr()
analyse.num_of_transactions()
analyse.community_detection()
analyse.topology_chars()
analyse.rich_club(normalized=False)
analyse.assortativity()

#agent = 0
#tracker.trans_time_visualizer(agent,'Transaction Time Tracker')
tracker.valuability()

for prop in ['money','asset','approval']:
    tracker.property_evolution(prop)

tracker.correlation_growth_situation('money','situation')
tracker.correlation_growth_situation('asset','situation')
tracker.correlation_growth_situation('approval','situation')

tracker.correlation_growth_situation('money','initial_money')
tracker.correlation_growth_situation('asset','initial_asset')
tracker.correlation_growth_situation('approval','initial_approval')

tracker.plot('self_value',title='Self Value')
tracker.plot('valuable_to_others',title='How Much Valuable to Others')
tracker.plot('worth_ratio',title='Worth_ratio Evolution by Time',alpha=1)
tracker.plot('correlation_mon',title='Correlation of Money and Situation')
tracker.plot('correlation_situ',title="Correlation of Situation and Neighbor's Situation")


fig, ax = plt.subplots(nrows=1,ncols=1)
probability = analyse.array('probability')
im = ax.imshow(probability.astype(float),aspect='auto',animated=True)
def animate(alpha):
    probability[probability < alpha/N] = 0
    im.set_array(probability)
    return im,
anim = animation.FuncAnimation(fig,animate,frames=20, interval=1000, blit=True)
anim.save(path+'probability.gif', writer='imagemagick')
plt.close()

analyse.graph_related_chars(num_transaction_tot,tracker)

""" community isomorphism investigation"""
analyse.community_detection()

for prop in ['money','asset','approval','worth_ratio']:
    analyse.communities_property_hist(prop)
    
plt.close('all')

