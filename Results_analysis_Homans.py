"""
Created on Thu Oct  3 12:04:52 2019
@author: Taha Enayat, Mohsen Mehrani

This file generates results created in Homans.py

To run, first specify N, T, and version of the file you want to analyse (line 23-25),

then run Homans.py for few seconds (let it run till numbers appear, then abort Homans.py) 
so that Agent class is known and can analyse the data.

Note that the project can run on Windows as well as Linux.
It creates a folder named 'runned_files' in directory which the code is and 
the data stores there.
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
import winsound
from datetime import datetime
start_time = datetime.now()

#XXX
N = 100 
T = 2000
version = '99.01.08_1 basic'

pd = {'win32':'\\', 'linux':'/'}
if sys.platform.startswith('win32'):
    plat = 'win32'
elif sys.platform.startswith('linux'):
    plat = 'linux'
current_path = os.getcwd()
path = current_path +pd[plat]+'runned_files'+pd[plat]+'N%d_T%d'%(N,T)+pd[plat]+version+pd[plat]

with open(path + 'Initials.txt','r') as initf:
    init_lines = initf.readlines()
    for i,line in enumerate(init_lines):
        if i == 2: sampling_time = int(line[:-1])
memory_size = 10
saving_time = T
path += '0_%d'%(T)+pd[plat]

"""Open File"""
with open(path+'Other_data.pkl','rb') as data_file:
    num_transaction_tot = pickle.load(data_file)
    explore_prob_arr    = pickle.load(data_file)
    rejection_agent     = pickle.load(data_file)
    
with open(path+'Agents.pkl','rb') as agent_file:
    A = pickle.load(agent_file)

with open(path + 'Tracker.pkl','rb') as tracker_file:
    tracker = pickle.load(tracker_file)

""" Plots""" 
analyse = Analysis_Tools_Homans.Analysis(N,T,memory_size,A,path)
main_graph = analyse.graph_construction('trans_number',num_transaction_tot, sampling_time=sampling_time, sample_time_trans=tracker.sample_time_trans)

for nsize in ['asset','money','approval','degree']:
    for ncolor in ['community','situation','worth_ratio']:
        for position in ['spring','kamada_kawai']:
            analyse.draw_graph_weighted_colored(position=position,nsize=nsize,ncolor=ncolor)

analyse.graph_correlations(all_nodes = False)
analyse.graph_correlations(all_nodes = True)

tracker.get_path(path) #essential
tracker.valuability()
tracker.plot_general(num_transaction_tot,title='Number of Transaction')
tracker.plot_general(explore_prob_arr * N,title='Average Exploration Probability',explore=True,N=N)

analyse.hist('degree')
analyse.hist_log_log('degree')
analyse.hist_log_log('neighbor',semilog=True)

i=0
array = np.zeros((8,N,N))
for prop in ['money','approval','asset','active_neighbor','utility','probability','neighbor','value']:
    analyse.hist(prop)
    analyse.hist_log_log(prop)
    array[i] = analyse.array(prop)
    i += 1

analyse.money_vs_situation(path+'money_vs_situation')
analyse.transaction_vs_property('asset')
analyse.transaction_vs_property('money')
analyse.transaction_vs_property('approval')
analyse.num_of_transactions()
analyse.community_detection()
analyse.topology_chars()
analyse.assortativity()
analyse.property_variation()
analyse.intercommunity_links()
analyse.prob_nei_correlation()
try:
    analyse.rich_club(normalized=True)
except: print('could not create rich club')

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

# tracker.correlation_pairplots(all_nodes = True)
# tracker.correlation_pairplots(present_nodes = main_graph.nodes())

tracker.correlation_pairplots()
tracker.correlation_pairplots(nodes_selection = 'graph_nodes', present_nodes = main_graph.nodes())

community_members = [list(x) for x in analyse.modularity_communities]
for num,com in enumerate(community_members):
    tracker.correlation_pairplots(nodes_selection ='community_nodes_{}'.format(num),present_nodes = com)


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
    analyse.communities_property_dist(prop)
#    analyse.communities_property_evolution(tracker,prop)

all_data_dict = analyse.graph_related_chars(num_transaction_tot,tracker,sampling_time)
analyse.path = path

tracker.rejection_history()

plt.close('all')
print (datetime.now() - start_time)
winsound.Beep(2000,500)