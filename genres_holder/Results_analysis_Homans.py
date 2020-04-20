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
# import winsound
from datetime import datetime
from progress.bar import Bar

start_time = datetime.now()

#XXX
N = 100
# T = 5000
# version = 'Result_Homans_1_b_2'

class Agent():
    def __init__(self,money,approval,situation):
        self.money = money
        self.approval = approval
        self.neighbor = np.zeros(N,dtype=int) #number of interactions
        self.value = np.full((N,memory_size),-1,dtype=float)
        self.time = np.full((N,memory_size),-1)
        self.situation = situation
        self.active_neighbor = {} #dictianary; keys are active neighbor indexes; values are probabilities
        self.sigma = Decimal('0') #sum of probabilities. used in normalization
        self.feeling = np.zeros(N)
        self.worth_ratio = self.approval/self.money
        self.asset = self.money + self.approval / self.worth_ratio
        return
    
    def asset_updater(self):
        self.asset = self.money + self.approval / self.worth_ratio
        return 
    
    def neighbor_average(self):
        
        self.n_avg = {'money':0, 'approval':0}
        for j in self.active_neighbor.keys():
            self.n_avg['money'] += A[j].money
            self.n_avg['approval'] += A[j].approval
        # self.n_avg['money'] = self.n_avg['money'] / len(self.active_neighbor)
        # self.n_avg['approval'] = self.n_avg['approval'] / len(self.active_neighbor)

        self.n_avg['money'] += self.money
        self.n_avg['approval'] += self.approval
        self.n_avg['money'] = self.n_avg['money'] / (len(self.active_neighbor)+1)
        self.n_avg['approval'] = self.n_avg['approval'] / (len(self.active_neighbor)+1)
        
        self.n_average = self.n_avg['approval'] / self.n_avg['money']
        return self.n_average

    def probability(self,neighbor,t):
        '''
        calculates probability for choosing each neighbor
        utility = value * acceptance_probability
        converts value to probability (normalized)
        uses proposition_3_and_4
        should be a list with the same size as of neighbors with numbers 0< <1
        '''
        
        if self.neighbor[neighbor] < memory_size:
            where = self.neighbor[neighbor]-1 #last value in memory
        else:
            where = memory_size-1
        
        p0 = np.exp(self.value[neighbor,where] * prob0_magnify_factor)
        p1 = self.frequency_to_probability(neighbor,t) * prob1_magnify_factor - (prob1_magnify_factor -1)
        p2 = np.exp(self.feeling[neighbor]) * prob2_magnify_factor - (prob2_magnify_factor -1)

        # p0 = 1.0
        # p1 = 1.0
        # p2 = 1.0
        
        p0_tracker.append(p0)
        p1_tracker.append(p1)
        p2_tracker.append(p2)
        
        probability = p0 * p1 * p2 #not normalized. normalization occurs in neighbor_concatenation()
        return Decimal(probability).quantize(Decimal('1e-5'),rounding = ROUND_DOWN) if probability < 10**8 else Decimal(10**8)
    
    def frequency_to_probability(self,neighbor,t):
        
        mask = (self.time[neighbor] > t-10) & (self.time[neighbor] != -1)
        n1 = np.size(self.time[neighbor][mask])
        short_term = 1 - alpha * (n1/10)
        n2 = self.neighbor[neighbor]
        long_term = 1 + beta * (n2 * len(self.active_neighbor) /(t*np.average(num_transaction_tot[:t-1]) ) ) 
        prob = short_term * long_term
        return prob
    
    
    def neighbor_concatenation(self,self_index,new_neighbor,t):
        sum_before = sum(list(self.active_neighbor.values()))
        sigma_before = self.sigma
        
        for j in self.active_neighbor.keys():
            self.active_neighbor[j] *= self.sigma
            
        grade_new_neighbor = self.probability(new_neighbor,t)

        if new_neighbor in self.active_neighbor:
            self.sigma += grade_new_neighbor - self.active_neighbor[new_neighbor]
        else:
            self.sigma += grade_new_neighbor
            
        self.active_neighbor[new_neighbor] = grade_new_neighbor
        
        sum_middle = sum(list(self.active_neighbor.values()))
        for j in self.active_neighbor.keys():
            if j!=new_neighbor:
                self.active_neighbor[j] /= self.sigma
                self.active_neighbor[j] = Decimal( str(self.active_neighbor[j]) ).quantize(Decimal('1e-5'),rounding = ROUND_DOWN)
                
        if new_neighbor in self.active_neighbor:
            self.active_neighbor[new_neighbor] = 1 - ( sum(self.active_neighbor.values()) -  self.active_neighbor[new_neighbor])
        else:
            self.active_neighbor[new_neighbor] = 1 -  sum(self.active_neighbor.values()) 
            
        #error finding
        if self.active_neighbor[new_neighbor] < 0:
            raise NegativeProbability('self index:',self_index,'neighbor',new_neighbor)
        elif np.size(np.array(list(self.active_neighbor.values()))[np.array(list(self.active_neighbor.values()))>1]) != 0:
            print('\nerror')
            print('self index',self_index)
            print('neighbor index',new_neighbor)
            print('sum after',sum(list(self.active_neighbor.values())))
            print('sum middle',sum_middle)
            print('sum before',sum_before)
            print('sigma before',sigma_before)
            print('sigma after',self.sigma)
            print('value',self.value[new_neighbor])
            print('intered prob',probability_new_neighbor)
            raise NegativeProbability('self index:',self_index,'neighbor',new_neighbor)
        elif sum(list(self.active_neighbor.values())) > 1.01 or sum(list(self.active_neighbor.values())) < 0.99:
            raise NegativeProbability('not one',sum(list(self.active_neighbor.values())))

        return

    def second_agent(self,self_index,self_active_neighbor):
        """returns an agent in memory with maximum utility to intract with
        probability() is like value
        proposition 6"""
        
        i = 0
        Max = 0
        for j in self_active_neighbor:
            probability = self.active_neighbor[j]
            other_probability = A[j].active_neighbor[self_index]
            utility = probability * other_probability
            if utility >= Max:
                Max = utility
                chosen_agent = j
                chosen_agent_index = i
            i += 1
        return chosen_agent , chosen_agent_index    
# =============================================================================

bar = Bar('Processing', max=14)


pd = {'win32':'\\', 'linux':'/'}
if sys.platform.startswith('win32'):
    plat = 'win32'
elif sys.platform.startswith('linux'):
    plat = 'linux'
current_path = os.getcwd()
path = current_path +pd[plat]+'runned_files'+pd[plat]+'N%d_T%d'%(N,T)+pd[plat]+version+pd[plat]

# with open(path + 'Initials.txt','r') as initf:
#     init_lines = initf.readlines()
#     for i,line in enumerate(init_lines):
#         if i == 2: sampling_time = int(line[:-1])
sampling_time = 1000
memory_size = 10
saving_time = T
path += '0_%d'%(T)+pd[plat]
# path = os.path.join(current_path, 'runned_files', 'N%d_T%d'%(N,T),version,'0_%d'%(T))
"""Open File"""
with open(os.path.join(path,'Other_data.pkl'),'rb') as data_file:
    num_transaction_tot = pickle.load(data_file)
    explore_prob_arr    = pickle.load(data_file)
    rejection_agent     = pickle.load(data_file)
    
with open(os.path.join(path,'Agents.pkl'),'rb') as agent_file:
    A = pickle.load(agent_file)

with open(os.path.join(path,'Tracker.pkl') ,'rb') as tracker_file:
    tracker = pickle.load(tracker_file)

""" Plots""" 
analyse = Analysis_Tools_Homans.Analysis(N,T,memory_size,A,path)
main_graph = analyse.graph_construction('trans_number',num_transaction_tot, sampling_time=sampling_time, sample_time_trans=tracker.sample_time_trans)

# for nsize in ['asset','money','approval','degree']:
#     for ncolor in ['community','situation','worth_ratio']:
#         for position in ['spring','kamada_kawai']:
#             analyse.draw_graph_weighted_colored(position=position,nsize=nsize,ncolor=ncolor)


bar.next() #XXX
analyse.graph_correlations(all_nodes = False)
analyse.graph_correlations(all_nodes = True)

bar.next() #XXX
tracker.get_path(path) #essential
tracker.plot_general(num_transaction_tot,title='Number of Transaction')
tracker.plot_general(explore_prob_arr * N,title='Average Exploration Probability',explore=True,N=N)

bar.next() #XXX
analyse.hist('degree')
analyse.hist_log_log('degree')
analyse.hist_log_log('neighbor',semilog=True)

bar.next() #XXX
i=0
array = np.zeros((8,N,N))
for prop in ['money','approval','asset','active_neighbor','utility','probability','neighbor','value']:
    analyse.hist(prop)
    analyse.hist_log_log(prop)
    array[i] = analyse.array(prop)
    i += 1

bar.next() #XXX
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

bar.next() #XXX
size = 10 
for rand_agent in np.random.choice(np.arange(N),size=size,replace=False):
    agent = rand_agent
    tracker.trans_time_visualizer(agent,'Transaction Time Tracker')
tracker.valuability()

for prop in ['money','asset','approval']:
    tracker.property_evolution(prop)

bar.next() #XXX
tracker.correlation_growth_situation('money','situation')
tracker.correlation_growth_situation('asset','situation')
tracker.correlation_growth_situation('approval','situation')

tracker.correlation_growth_situation('money','initial_money')
tracker.correlation_growth_situation('asset','initial_asset')
tracker.correlation_growth_situation('approval','initial_approval')
plt.close('all')

bar.next() #XXX
tracker.plot('self_value',title='Self Value')
tracker.plot('valuable_to_others',title='How Much Valuable to Others')
tracker.plot('worth_ratio',title='Worth_ratio Evolution by Time',alpha=1)
tracker.plot('correlation_mon',title='Correlation of Money and Situation')
tracker.plot('correlation_situ',title="Correlation of Situation and Neighbor's Situation")

# tracker.correlation_pairplots(all_nodes = True)
# tracker.correlation_pairplots(present_nodes = main_graph.nodes())

bar.next() #XXX
tracker.correlation_pairplots()
tracker.correlation_pairplots(nodes_selection = 'graph_nodes', present_nodes = main_graph.nodes())

bar.next() #XXX
community_members = [list(x) for x in analyse.modularity_communities]
for num,com in enumerate(community_members):
    tracker.correlation_pairplots(nodes_selection ='community_nodes_{}'.format(num),present_nodes = com)

bar.next() #XXX
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

bar.next() #XXX
for prop in ['money','asset','approval','worth_ratio']:
    analyse.communities_property_dist(prop)
#    analyse.communities_property_evolution(tracker,prop)

bar.next() #XXX
all_data_dict = analyse.graph_related_chars(num_transaction_tot,tracker,sampling_time)
analyse.path = path

bar.next() #XXX
tracker.rejection_history()

plt.close('all')
print (datetime.now() - start_time)
bar.finish()
# winsound.Beep(2000,500)
