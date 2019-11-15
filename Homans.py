"""
Created on Mon Aug 12 10:12:03 2019
@author: Taha Enayat, Mohsen Mehrani
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#import networkx as nx
import winsound
import pickle
import Analysis_Tools_Homans
import os
import matplotlib.animation as animation
import shutil


start_time = datetime.now()

# =============================================================================
"""Class"""

class NegativeProbability(Exception):
    pass

class Agent():
    def __init__(self,money,approval,situation):
        self.money = money
        self.approval = approval
        self.neighbor = np.zeros(N,dtype=int) #number of interactions
        self.value = np.full((N,memory_size),-1,dtype=float)
        self.time = np.full((N,memory_size),-1)
        self.situation = situation
        self.active_neighbor = {} #dictianary; keys are active neighbor indexes; values are probabilities
        self.sigma = 0 #sum of probabilities. used in normalization
        self.feeling = np.zeros(N)
        self.worth_ratio = self.approval/self.money
        self.asset = self.money + self.approval / self.worth_ratio
        return
    
    def average_worth_ratio(self):
        worth_ratio_avg = 0
        for agent in A:
            worth_ratio_avg += agent.worth_ratio 
        worth_ratio_avg/= N
        return worth_ratio_avg
    
    def asset_updater(self):
#        self.asset = self.money + self.approval / self.worth_ratio
        self.asset = self.money + self.approval / self.average_worth_ratio()
        return 
    
    def neighbor_average(self):
        
        self.n_avg = {'money':0, 'approval':0}
        for j in self.active_neighbor.keys():
            self.n_avg['money'] += A[j].money
            self.n_avg['approval'] += A[j].approval
        self.n_avg['money'] = self.n_avg['money'] / len(self.active_neighbor)
        self.n_avg['approval'] = self.n_avg['approval'] / len(self.active_neighbor)

#        self.n_avg['money'] += self.money
#        self.n_avg['approval'] += self.approval
#        self.n_avg['money'] = self.n_avg['money'] / (len(self.active_neighbor)+1)
#        self.n_avg['approval'] = self.n_avg['approval'] / (len(self.active_neighbor)+1)
        
#        self.n_average = self.n_avg['approval'] / self.n_avg['money']
        self.n_average = zeta * self.worth_ratio + (1 - zeta)*self.n_avg['approval'] / self.n_avg['money']
        return self.n_average

#    def neighbor_average(self):
#        
#        self.n_avg = 0
#        for j in self.active_neighbor.keys():
#            self.n_avg += A[j].approval / A[j].money
#        self.n_avg += self.approval / self.money
#        self.n_avg /= len(self.active_neighbor) + 1
#        return self.n_avg


#    def neighbor_average(self):
#        
#        worth = 0
#        for k in self.active_neighbor.keys():
#            worth += A[k].worth_ratio
#        worth /= len(self.active_neighbor)
#        return worth
    
    def probability(self,neighbor,t):
        '''calculates probability for choosing each neighbor
        utility = value * acceptance_probability
        converts value to probability (normalized)
        uses proposition_3_and_4
        should be a list with the same size as of neighbors with numbers 0< <1'''
        
        if self.neighbor[neighbor] < memory_size:
            where = self.neighbor[neighbor]-1 #last value in memory
        else:
            where = memory_size-1
        
        value = self.value[neighbor,where]
#        p0 = np.arctan(factor*value)*2/np.pi + 1 #ranges from 0 to 2: value=0 maps to p=1. that means p=1 is the defaul number.
#        p0 = np.arctan(prob0_magnify_factor*value*10)*2/np.pi + 1 #ranges from 0 to 2: value=0 maps to p=1. that means p=1 is the defaul number.
#        p0 = np.exp(value * prob0_magnify_factor)
#        p0 = value * prob0_magnify_factor 
        p0 = value * prob0_magnify_factor + 1
        p1 = self.frequency_to_probability(neighbor,t) * prob1_magnify_factor - (prob1_magnify_factor -1)
        p2 = np.exp(self.feeling[neighbor]) * prob2_magnify_factor - (prob2_magnify_factor -1)
#        p1 = 1.0
        
        p0_tracker.append(p0)
        p1_tracker.append(p1)
        p2_tracker.append(p2)
        
        probability = p0 * p1 * p2 #not normalized. normalization occurs in neighbor_concatenation()
        return probability
    
    def frequency_to_probability(self,neighbor,t):
        mask = (self.time[neighbor] > t-10) & (self.time[neighbor] != -1)
        n1 = np.size(self.time[neighbor][mask])
        short_term = 1 - alpha * (n1/10)
        n2 = self.neighbor[neighbor]
        long_term = 1 + beta * (n2 * len(self.active_neighbor) /t) 
        prob = short_term * long_term
        return prob
    
    
    def neighbor_concatenation(self,self_index,new_neighbor,t):
        sum_before = sum(list(self.active_neighbor.values()))
        for j in self.active_neighbor.keys():
            self.active_neighbor[j] *= self.sigma
        
        sigma_before = self.sigma            
        probability_new_neighbor = self.probability(new_neighbor,t)
        sum_middle = sum(list(self.active_neighbor.values()))

        if new_neighbor in self.active_neighbor:
            self.sigma += probability_new_neighbor - self.active_neighbor[new_neighbor]
        else:
            self.sigma += probability_new_neighbor
            
#        before = np.zeros(len(self.active_neighbor))
#        after = np.zeros(len(self.active_neighbor))
#        i = 0
        self.active_neighbor[new_neighbor] = probability_new_neighbor
        for j in self.active_neighbor.keys():
#            before[i] = self.active_neighbor[j]
            self.active_neighbor[j] /= self.sigma
#            after[i] = self.active_neighbor[j]
#            i += 1
        if np.size(np.array(list(self.active_neighbor.values()))[np.array(list(self.active_neighbor.values()))>1]) != 0:
            #normalize again
            summ = sum(self.active_neighbor.values())
            for j in self.active_neighbor:
                self.active_neighbor[j]/summ

        #error finding
        if probability_new_neighbor < 0:
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
#            print('active before',active_before)
#            print('active middle',active_middle)
#            print('active after',self.active_neighbor)
#            for i in np.arange(np.size(before)):
#                print('b',before[i],'a',after[i])
            raise NegativeProbability('self index:',self_index,'neighbor',new_neighbor)
        elif sum(list(self.active_neighbor.values())) > 1.01 or sum(list(self.active_neighbor.values())) < 0.99:
            raise NegativeProbability('not one',sum(list(self.active_neighbor.values())))
#        elif self.sigma > len(self.active_neighbor):
#            raise NegativeProbability('sigma error')


        return

    def second_agent(self,self_index,self_active_neighbor):
        """returns an agent in memory with maximum utility to intract with
        probability() is like value
        proposition 6"""
        
#        self_active_neighbor = self.active_neighbor.keys()
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

"""Functions"""
def transaction(index1,index2,t):
    
    agent1 = A[index1]
    agent2 = A[index2]
    number_of_transaction1 = agent1.neighbor[index2]
    number_of_transaction2 = agent2.neighbor[index1]

    if index1 in agent2.active_neighbor:
        p = agent2.active_neighbor[index1]
        acceptance_util = np.random.choice([0,1],p=[1-p,p])
    else:
        acceptance_util = 1

    if len(agent1.active_neighbor) != 0:
        worth_ratio1 = agent1.neighbor_average()
    else:
        worth_ratio1 = agent1.worth_ratio
    if len(agent2.active_neighbor) != 0:
        worth_ratio2 = agent2.neighbor_average()
    else:
        worth_ratio2 = agent2.worth_ratio

    if agent2.approval > 0.001:
        if worth_ratio2 >= worth_ratio1:
            acceptance_worth = 1
        else:
    #        acceptance_worth = 0
            p = np.exp( -(worth_ratio1 - worth_ratio2)/normalization_factor )
            acceptance_worth = np.random.choice([0,1],p=[1-p,p])
    else:
        acceptance_worth = 0
    
    p = np.exp( -np.abs(agent1.asset - agent2.asset)/param )
    acceptance_asset = np.random.choice([0,1],p=[1-p,p])
    
    amount = transaction_percentage * agent1.money
    agreement_point = (worth_ratio2 - worth_ratio1)/(worth_ratio2 + worth_ratio1) * amount * worth_ratio1 #x=(E2-E1/E2+E1)*AE1
    threshold = threshold_percentage[index2] * agent2.approval

    if threshold > (amount * worth_ratio1 + agreement_point):
        acceptance_thr = 1
    else: acceptance_thr = 0
    
    acceptance_tracker[t-1] += acceptance_thr
    
    acceptance = acceptance_worth * acceptance_thr * acceptance_asset * acceptance_util
    if acceptance:   #transaction accepts
        num_transaction_tot[t-1] += 1
        agreement_tracker.append(agreement_point)
        
        feeling = agreement_point / worth_ratio1 #is equal for both (from definition)

#        value1 = (agreement_point / worth_ratio1) / agent1.money #normalized to property: does not depend on wealth
#        value2 = (agreement_point / worth_ratio1) / agent2.money #practically it is: (E2-E1/E2+E1)*percentage
        #not good. same worth_ratios lead to value=0 which is like when value is negative (in probability function it is interpreted as zero probability)
        value1 = + amount + agreement_point/worth_ratio1
        value2 = + amount
        #let's see whether this one works well

        agent1.neighbor[index2] += 1
        agent2.neighbor[index1] += 1
        agent1.feeling[index2] = feeling
        agent2.feeling[index1] = feeling

        #doing the transaction
        agent1.money -= np.round(amount,3)
        agent2.money += np.round(amount,3)
        agent1.approval += np.round(amount*worth_ratio1 + agreement_point,3)
        agent2.approval -= np.round(amount*worth_ratio1 + agreement_point,3)
        
        
#        agent1.worth_ratio = (amount*worth_ratio1 + agreement_point) / amount # = approval/money
#        agent2.worth_ratio = (amount*worth_ratio1 + agreement_point) / amount # eqaul for both.
        agent1.worth_ratio = lamda * agent1.worth_ratio + (1-lamda) * (amount*worth_ratio1 + agreement_point) / amount
        agent2.worth_ratio = lamda * agent2.worth_ratio + (1-lamda) * (amount*worth_ratio1 + agreement_point) / amount

        
        agent1.asset_updater()
        agent2.asset_updater()
        asset_tracker[index1].append(agent1.asset)
        asset_tracker[index2].append(agent2.asset)
        
        #changing the memory        
        if number_of_transaction1 < memory_size:#memory is not full
            empty_memory = number_of_transaction1
            agent1.time [index2,empty_memory] = t
            agent1.value[index2,empty_memory] = value1
        else:
            shift_memory( agent1 , index2)
            agent1.time [index2,memory_size-1] = t
            agent1.value[index2,memory_size-1] = value1
            
        if number_of_transaction2 < memory_size:#memory is not full
            empty_memory = number_of_transaction2
            agent2.time [index1,empty_memory] = t
            agent2.value[index1,empty_memory] = value2
        else:
            shift_memory(agent2,index1)
            agent2.time [index1,memory_size-1] = t
            agent2.value[index1,memory_size-1] = value2

        agent1.neighbor_concatenation(index1,index2,t)
        agent2.neighbor_concatenation(index2,index1,t)
        
    return acceptance

# =============================================================================
def explore(index,t):
    '''choose another agent which is not in his memory
    uses proposition_2 (similar situation)
    before calling this function we have to check if the transaction was rewarding
    something have to be done in case there is no neighbors
    repetitious neighbors should be avoided'''
    
    agent = A[index]
    mask = np.ones(N,dtype=bool)
    mask[index] = False
    agent_active_neighbor = list(agent.active_neighbor.keys())
    self_similarity = agent.situation
    
    num_explore[t-1] += 1
    
    if len(agent_active_neighbor) != N-1:
        if len(agent_active_neighbor) != 0:
            chosen_neighbor_index = np.random.choice(agent_active_neighbor,p=list(agent.active_neighbor.values()))
            situation = A[chosen_neighbor_index].situation
            mask[agent_active_neighbor] = False
            if np.size(mask[mask==True]) >= num_of_tries:
                arri_choice = np.random.choice(np.arange(N)[mask],size=num_of_tries,replace=False)
            else:
                num_true_in_mask = np.size(mask[mask==True])
                arri_choice = np.random.choice(np.arange(N)[mask],size=num_true_in_mask,replace=False)
            
            for other_index in arri_choice:
                other_situation = A[other_index].situation
#                p = np.exp(-np.abs(situation-other_situation)/similarity)
#                acceptance = np.random.choice([0,1],p=[1-p,p])
#                if acceptance == 1:
                if other_situation > (situation-similarity) and other_situation < (situation+similarity):
                    
                    other_agent = A[other_index]
                    if len(other_agent.active_neighbor) != 0:
                        
                        nearest_choice = 1 #maximum possible situation difference
                        for k in other_agent.active_neighbor.keys():
                            diff_abs = np.abs(A[k].situation - self_similarity)
                            if diff_abs < nearest_choice:
                                nearest_choice = diff_abs
                                nearest_choice_index = k
                        p = other_agent.active_neighbor[nearest_choice_index]
                        acceptance = np.random.choice([0,1],p=[1-p,p])

#                        acceptance = 1
                        if acceptance == 1:
                            transaction(index,other_index,t)
                    else:
                        transaction(index,other_index,t)
                            
                    if other_index in agent.active_neighbor:  #which means transaction has been accepted
                        similarity_tracker[index].append(other_situation)
                        break
        else:
            other_index = np.random.choice(np.arange(N)[mask])
            other_agent = A[other_index]
            other_situation = other_agent.situation
            if len(other_agent.active_neighbor) != 0:

                nearest_choice = 1 #maximum possible situation difference
                for k in other_agent.active_neighbor.keys():
                    diff_abs = np.abs(A[k].situation - self_similarity)
                    if diff_abs < nearest_choice:
                        nearest_choice = diff_abs
                        nearest_choice_index = k
                p = other_agent.active_neighbor[nearest_choice_index]
                acceptance = np.random.choice([0,1],p=[1-p,p])

#                acceptance = 1
                if acceptance == 1:
                    transaction(index,other_index,t)
            else:
                transaction(index,other_index,t)
            
            if other_index in agent.active_neighbor:
                similarity_tracker[index].append(A[other_index].situation)
    return

# =============================================================================
def shift_memory(agent,index):
    temp = np.delete(agent.value[index],0)
    agent.value[index] = np.concatenate((temp,[-1]))
    temp = np.delete(agent.time[index],0)
    agent.time[index] = np.concatenate((temp,[-1]))
    return
# =============================================================================
def save_it(version):
    current_path = os.getcwd()

    try:
        os.mkdir(current_path+'\\runned_files')
    except OSError:
        print ("runned_files already exists")
        
    try:
        os.mkdir(current_path+'\\runned_files'+'\\N%d_T%d'%(N,T))
    except OSError:
        print ("version already exists")
        
    try:
        os.mkdir(current_path+'\\runned_files'+'\\N%d_T%d\\'%(N,T)+version)
    except OSError:
        print ("Creation of the directory failed")
    
    path = 'runned_files'+'\\N%d_T%d\\'%(N,T)+version+'\\'
#    np.save(path+'Agents.npy',A)
#    np.save(path+'Tracker.npy',tracker)
    with open(path + 'Agents.pkl','wb') as agent_file:
        pickle.dump(A,agent_file,pickle.HIGHEST_PROTOCOL)

#    with open(path + 'Tracker.pkl','wb') as tracker_file:
#        pickle.dump(tracker,tracker_file,pickle.HIGHEST_PROTOCOL)
        
    return path
# =============================================================================
"""Parameters"""#XXX

N = 100
T = 10*N
similarity = 0.05                   #how much this should be?
memory_size = 10                    #contains the last memory_size number of transaction times
transaction_percentage = 0.1        #percent of amount of money the first agent proposes from his asset 
num_of_tries = 20                   #in function explore()
threshold_percentage =np.full(N,1)#the maximum amount which the agent is willing to give
normalization_factor = 1            #used in transaction(). what should be?
prob0_magnify_factor = 0.3          #this is in probability() for changing value so that it can take advantage of arctan
prob1_magnify_factor = 2
prob2_magnify_factor = 1
alpha = 1                           #in short-term effect of the frequency of transaction
beta = 0.3                          #in long-term effect of the frequency of transaction
param = 2                           #a normalizing factor in assigning the acceptance probability. It normalizes difference of money of both sides
lamda = 0                           # how much one agent relies on his last worth_ratio and how much relies on current transaction's worth_ratio
zeta = 0.8
"""Initial Condition"""

situation_arr = np.random.random(N) #randomly distributed
#money = np.full(N,2)
#money = np.round(np.random.normal(loc=5.5,scale=1,size=N),decimals=3)
#money = 1 + situation_arr * 2
#money = np.zeros(N)
money = np.round(np.random.rand(N) * 9 + 1 ,decimals=3)
#money = np.round(situation_arr[:] * 9 + 1 ,decimals=3)
approval = np.full(N,5.5)
#approval = 1 + situation_arr * 2
#approval = np.round(situation_arr[:] * 9 + 1 ,decimals=3)
#approval = np.round(11 - money[:],decimals=3)
#risk_receptibility = np.random.random(N)*4*similarity

A = np.zeros(N,dtype=object)
explore_prob_array = np.zeros(T)
for i in np.arange(N):
    A[i]=Agent( money[i], approval[i], situation_arr[i]) 

"""trackers"""
tracker = Analysis_Tools_Homans.Tracker(N,T,memory_size,A)
num_transaction_tot = np.zeros(T)
num_explore = np.zeros(T)
agreement_tracker = []
acceptance_tracker = np.zeros(T)
p0_tracker = []
p1_tracker = []
p2_tracker = []
similarity_tracker = [ [] for _ in np.arange(N) ]
asset_tracker = [ [] for _ in np.arange(N) ]
# =============================================================================
"""Main"""

"""choose one agent
find another agent through calculating probability
explores for new agent (expands his memory)"""

for t in np.arange(T)+1:#t goes from 1 to T
    """computations"""
#    situation_arr = np.copy(money)
    print(t)
    shuffled_agents=np.arange(N)
    np.random.shuffle(shuffled_agents)
    for i in shuffled_agents:
        person = A[i]
        person_active_neighbor_size = len(person.active_neighbor)
        exploration_probability = (N-1-person_active_neighbor_size)/(N-1)#(2*N-2)
        explore_prob_array[t-1] += exploration_probability
        if person_active_neighbor_size != 0: #memory is not empty
            rand = np.random.choice([1,0],size=1,p=[1-exploration_probability,exploration_probability])
            if rand==1:
                
                person_active_neighbor = np.array(list(person.active_neighbor.keys()))
                if person_active_neighbor_size < num_of_tries:
                    num_of_choice = person_active_neighbor_size
                else:
                    num_of_choice = num_of_tries
                choice_arr = np.zeros(num_of_choice,dtype=int)
                for k in np.arange(num_of_choice):
                    choice_arr[k] , chosen_index = person.second_agent(i,person_active_neighbor)
                    person_active_neighbor = np.delete(person_active_neighbor, chosen_index)
                
                for j in choice_arr:
                    if transaction(i,j,t):
                        break
            else:
                explore(i,t)
        else:
            explore(i,t)
    
    """trackers"""
    tracker.update_A(A)
    tracker.get_list('self_value',t-1)
    tracker.get_list('valuable_to_others',t-1)
    tracker.get_list('trans_time',t-1)
    tracker.get_list('correlation_mon',t-1)
    tracker.get_list('correlation_situ',t-1)

    tracker.get_list('money',t-1)
    tracker.get_list('asset',t-1)
    
    if t>2:
        tracker.get_list('worth_ratio',t-3)
    explore_prob_array[t-1] /= N


print(datetime.now() - start_time)
# =============================================================================
"""Write File"""
version = 'avg_worth_ratio_self_importance' #XXX
path = save_it(version)
# =============================================================================
"""Analysis and Measurements"""
shutil.copyfile(os.getcwd()+'\\Homans.py',path+'\\Homans.py')
shutil.copyfile(os.getcwd()+'\\Analysis_Tools_Homans.py',path+'\\Analysis_Tools_Homans.py')

tracker.get_path(path)
analyse = Analysis_Tools_Homans.Analysis(N,T,memory_size,A,path,num_transaction_tot,explore_prob_array)
analyse.graph_construction('trans_number',num_transaction_tot,explore_prob_array,tracker_obj=tracker)
analyse.draw_graph_weighted_colored()
analyse.graph_correlations()

#a_money       = analyse.array('money')
#a_approval    = analyse.array('approval')
#a_worth_ratio = analyse.array('worth_ratio')
#a_neighbor    = analyse.array('neighbor')
#a_value       = analyse.array('value')
#a_time        = analyse.array('time')
#a_probability = analyse.array('probability')
#a_utility     = analyse.array('utility')
#a_situation   = analyse.array('situation')
#a_asset       = analyse.array('asset')

#analyse.hist('money')
#analyse.hist_log_log('money')
#analyse.hist('approval')
#analyse.hist_log_log('approval')
analyse.hist('degree')
#analyse.hist_log_log('degree')
#analyse.hist('value')
#analyse.hist_log_log('value')
analyse.hist('probability')
analyse.hist_log_log('probability')
#analyse.hist('utility')
#analyse.hist_log_log('utility')
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

"""tracker plots"""
agent = 0
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

#tracker.plot('self_value',title='Self Value')
#tracker.plot('valuable_to_others',title='How Much Valuable to Others')
tracker.plot('worth_ratio',title='Worth_ratio Evolution by Time',alpha=1)
tracker.plot('correlation_mon',title='Correlation of Money and Situation')
tracker.plot('correlation_situ',title="Correlation of Situation and Neighbor's Situation")
#tracker.trans_time_visualizer(3,'Transaction Time Tracker')
#
tracker.plot_general(num_transaction_tot, title='Number of Transaction Vs Time')
tracker.plot_general(num_explore, title='Number of Explorations Vs Time')
#tracker.plot_general(agreement_tracker, title='agreement Point')
#tracker.plot_general(acceptance_tracker, title='Threshold Acceptance Over Time')
#
plt.figure()
plt.plot(p0_tracker[::2])
plt.plot(p1_tracker[::2])
plt.plot(p2_tracker[::2])
plt.title('P0 & P1 & P2')
plt.savefig(path+'P0 & P1 & P2')
#tracker.hist_general(p0_tracker,title='p0')
#tracker.hist_general(p1_tracker,title='p1')
#tracker.hist_general(p2_tracker,title='p2')
#tracker.hist_log_log_general(p0_tracker,title='P0')
#tracker.hist_log_log_general(p1_tracker,title='P1')
#tracker.hist_log_log_general(p2_tracker,title='P2')

#plt.figure()
#for i in np.arange(N):
#    plt.plot(similarity_tracker[i])
#plt.ylim([0,1.1])
#plt.title('Similarity Tracker')

plt.figure()
for i in np.arange(N):
    plt.plot(asset_tracker[i])
plt.title('Asset Tracker')
plt.savefig(path+'Asset Tracker')



fig, ax = plt.subplots(nrows=1,ncols=1)
probability = analyse.array('probability')
im = ax.imshow(probability.astype(float),aspect='auto',animated=True)
def animate(alpha):
    probability[probability < alpha/N] = 0
    im.set_array(probability)
    return im,
anim = animation.FuncAnimation(fig,animate,frames=20, interval=1000, blit=True)
anim.save(path+'probability.gif', writer='imagemagick')

"""closes all figures"""
plt.close('all')

"""Time Evaluation"""
duration = 500  # millisecond
freq = 2000  # Hz
winsound.Beep(freq, duration)
print (datetime.now() - start_time)

analyse.graph_related_chars(num_transaction_tot,explore_prob_array,tracker) #make sure it is the last thing that runs
