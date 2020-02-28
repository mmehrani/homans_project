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
import sys

N = 50
T = 500
memory_size = 10
initial_time = 0
time_step = 500
sampling_time = 500
version = '1_basic_run'

pd = {'win32':'\\', 'linux':'/'}
if sys.platform.startswith('win32'):
    plat = 'win32'
elif sys.platform.startswith('linux'):
    plat = 'linux'
current_path = os.getcwd()
path = current_path +pd[plat]+'runned_files'+pd[plat]+'N%d_T%d'%(N,T)+pd[plat]+version+pd[plat]
path += '%d_%d'%(0, initial_time + time_step)+pd[plat]

"""Open File"""
with open(path+'Other_data.pkl','rb') as data_file:
    num_transaction_tot = pickle.load(data_file)
    explore_prob_arr    = pickle.load(data_file)
    rejection_agent     = pickle.load(data_file)
    
with open(path+'Agents.pkl','rb') as agent_file:
    a_matrix = pickle.load(agent_file)

with open(path + 'Tracker.pkl','rb') as tracker_file:
    tracker = pickle.load(tracker_file)
""""""
#tool = graph_tools_glossary.Graph_related_tools(1200,N,a_matrix)
#tool.graph_construction('trans_number',num_transaction_tot,sample_time_trans = tracker.sample_time_trans)
""" Plots""" 
analyse = Analysis_Tools_Homans.Analysis(N,T,memory_size,a_matrix,path)
main_graph = analyse.graph_construction('trans_number',num_transaction_tot, sampling_time=sampling_time, sample_time_trans=tracker.sample_time_trans)
analyse.draw_graph_weighted_colored()

analyse.graph_correlations(all_nodes = False)
analyse.graph_correlations(all_nodes = True)

#constructed_graph = analyse.G
#dynamic_graph = tracker.make_dynamic_trans_time_graph(constructed_graph)
#nx.write_gexf(dynamic_graph,path+'dynamic_trans_number_graph.gexf')

tracker.get_path(path) #essential
tracker.valuability()
#tracker.plot_general(explore_prob_arr * N,title='Average Exploration Probability')#,explore=True)
tracker.plot_general(num_transaction_tot,title='Number of Transaction')

analyse.hist('degree')
analyse.hist_log_log('degree')
i=0
array = np.zeros((7,N,N))
for prop in ['money','approval','asset','utility','probability','neighbor','value']:
    analyse.hist(prop)
    analyse.hist_log_log(prop)
    array[i] = analyse.array(prop)
    i += 1

analyse.money_vs_situation(path+'money_vs_situation')
analyse.transaction_vs_asset()
#analyse.degree_vs_attr()
analyse.num_of_transactions()
analyse.community_detection()
analyse.topology_chars()
analyse.rich_club(normalized=False)
analyse.assortativity()

#if initial_time == T - time_step:
if True:
    size = 10
    for rand_agent in np.random.choice(np.arange(N),size=size,replace=False):
        agent = rand_agent
        tracker.trans_time_visualizer(agent,'Transaction Time Tracker')
tracker.valuability()

for prop in ['money','asset','approval']:
    tracker.property_evolution(prop)

tracker.correlation_growth_situation('money','situation')
tracker.correlation_growth_situation('asset','situation')
tracker.correlation_growth_situation('approval','situation')

tracker.correlation_growth_situation('money','initial_money')
tracker.correlation_growth_situation('asset','initial_asset')
tracker.correlation_growth_situation('approval','initial_approval')
plt.close('all')

tracker.plot('self_value',title='Self Value')
tracker.plot('valuable_to_others',title='How Much Valuable to Others')
tracker.plot('worth_ratio',title='Worth_ratio Evolution by Time',alpha=1)
tracker.plot('correlation_mon',title='Correlation of Money and Situation')
tracker.plot('correlation_situ',title="Correlation of Situation and Neighbor's Situation")

tracker.correlation_pairplots(all_nodes = True)
tracker.correlation_pairplots(present_nodes = main_graph.nodes())

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

analyse.community_detection()

for prop in ['money','asset','approval','worth_ratio']:
    analyse.communities_property_hist(prop)
#    analyse.communities_property_evolution(tracker,prop)

analyse.graph_related_chars(num_transaction_tot,tracker,sampling_time)

plt.close('all')

