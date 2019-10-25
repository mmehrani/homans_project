"""
Created on Mon Aug 12 10:12:03 2019
@author: Taha Enayat
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import networkx as nx
import winsound
import pickle
import Analysis_Tools_Homans
import os

start_time = datetime.now()

# =============================================================================
"""Class"""

class Agent():
    def __init__(self,money,approval,situation):
        self.money = money
        self.approval = approval
        self.neighbour = np.zeros(N,dtype=int) #number of interactions
        self.value = np.full((N,memory_size),-1,dtype=float)
        self.time = np.full((N,memory_size),-1)
        self.situation = situation
        self.active_neighbour = {} #dictianary; keys are active neighbour indexes; values are probabilities
        self.sigma = 0 #sum of probabilities. used in normalization
        self.feeling = np.zeros(N)
        self.worth_ratio = self.approval/self.money
#        self.worth_ratio = 0
        return
    
    def property(self):
        return {'money':self.money, 'approval':self.approval}
    
    def neighbour_average(self):
        
        self.n_avg = {'money':0, 'approval':0}
        for j in self.active_neighbour.keys():
            self.n_avg['money'] += A[j].money
            self.n_avg['approval'] += A[j].approval
        self.n_avg['money'] = self.n_avg['money']/len(self.active_neighbour)
        self.n_avg['approval'] = self.n_avg['approval']/len(self.active_neighbour)
        self.n_average = self.n_avg['approval'] / self.n_avg['money']
        return self.n_average

#    def neighbour_average(self):
#        
#        self.n_avg = 0
#        for j in self.active_neighbour.keys():
#            self.n_avg += A[j].approval / A[j].money
#        self.n_avg /= len(self.active_neighbour)
#        return self.n_avg


#    def neighbour_average(self):
#        
#        self.worth = 0
#        for k in self.active_neighbour.keys():
#            self.worth += A[k].worth_ratio
#        self.worth /= len(self.active_neighbour)
#        return self.worth
    
    def probability(self,neighbour,t):
        '''calculates probability for choosing each neighbour
        utility = value * acceptance_probability
        converts value to probability (normalized)
        uses proposition_3_and_4
        should be a list with the same size as of neighbours with numbers 0< <1'''
        
        if self.neighbour[neighbour] < memory_size:
            where = self.neighbour[neighbour]-1 #last value in memory
        else:
            where = memory_size-1
        
        value = self.value[neighbour,where]
#        p0 = np.arctan(factor*value)*2/np.pi + 1 #ranges from 0 to 2: value=0 maps to p=1. that means p=1 is the defaul number.
#        p0 = np.arctan(prob0_magnify_factor*value*10)*2/np.pi + 1 #ranges from 0 to 2: value=0 maps to p=1. that means p=1 is the defaul number.
#        p0 = np.exp(value * prob0_magnify_factor)
#        p0 = value * prob0_magnify_factor 
        p0 = value * prob0_magnify_factor + 1
        p1 = self.frequency_to_probability(neighbour,t) * prob1_magnify_factor - (prob1_magnify_factor -1)
        p2 = np.exp(self.feeling[neighbour] * prob2_magnify_factor)
#        p1 = 1.0
        
        p0_tracker.append(p0)
        p1_tracker.append(p1)
        p2_tracker.append(p2)
        
        probability = p0 * p1 * p2 #not normalized. normalization occurs in neighbour_concatenation()
        return probability
    
    def frequency_to_probability(self,neighbour,t):
        mask = (self.time[neighbour] > t-10) & (self.time[neighbour] != -1)
        n1 = np.size(self.time[neighbour][mask])
        short_term = 1 - alpha * (n1/10)
        n2 = self.neighbour[neighbour]
        long_term = 1 + beta * (n2 * len(self.active_neighbour) /t) 
        prob = short_term * long_term
        return prob
    
    
    def neighbour_concatenation(self,self_index,new_neighbour,t):

        for j in self.active_neighbour.keys():
            self.active_neighbour[j] *= self.sigma
        
        probability_new_neighbour = self.probability(new_neighbour,t)
        if new_neighbour in self.active_neighbour:
            self.sigma += probability_new_neighbour - self.active_neighbour[new_neighbour]
        else:
            self.sigma += probability_new_neighbour
            
        self.active_neighbour[new_neighbour] = probability_new_neighbour
        for j in self.active_neighbour.keys():
            self.active_neighbour[j] /= self.sigma
        if np.size(np.array(list(self.active_neighbour.values()))[np.array(list(self.active_neighbour.values()))>1]) != 0:
            #normalize again
            summ = sum(self.active_neighbour.values())
            for j in self.active_neighbour:
                self.active_neighbour[j]/summ
        
        return

    def second_agent(self,self_index,self_active_neighbour):
        """returns an agent in memory with maximum utility to intract with
        probability() is like value
        proposition 6"""
        
#        self_active_neighbour = self.active_neighbour.keys()
        i = 0
        Max = 0
        for j in self_active_neighbour:
            probability = self.active_neighbour[j]
            other_probability = A[j].active_neighbour[self_index]
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
    number_of_transaction1 = agent1.neighbour[index2]
    number_of_transaction2 = agent2.neighbour[index1]

    if index1 in agent2.active_neighbour:
        p = agent2.active_neighbour[index1]
        acceptance_util = np.random.choice([0,1],p=[1-p,p])
    else:
        acceptance_util = 1

    if len(agent1.active_neighbour) != 0:
        worth_ratio1 = agent1.neighbour_average()
    else:
        worth_ratio1 = agent1.worth_ratio
    if len(agent2.active_neighbour) != 0:
        worth_ratio2 = agent2.neighbour_average()
    else:
        worth_ratio2 = agent2.worth_ratio

    
    if worth_ratio2 >= worth_ratio1:
        acceptance_worth = 1
    else:
#        acceptance_worth = 0
        p = np.exp( -(worth_ratio1 - worth_ratio2)/normalization_factor )
        acceptance_worth = np.random.choice([0,1],p=[1-p,p])
    
    p = np.exp( -np.abs(agent1.money - agent2.money)/param )
    acceptance_mon = np.random.choice([0,1],p=[1-p,p])
    
    amount = transaction_percentage * agent1.money
    agreement_point = (worth_ratio2 - worth_ratio1)/(worth_ratio2 + worth_ratio1) * amount * worth_ratio1 #x=(E2-E1/E2+E1)*AE1
    threshold = threshold_percentage[index2] * agent2.approval

    if threshold > (amount * worth_ratio1 + agreement_point):
        acceptance_thr = 1
    else: acceptance_thr = 0
    
    acceptance_tracker[t-1] += acceptance_thr
    
    acceptance = acceptance_worth * acceptance_thr * acceptance_mon * acceptance_util
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

        agent1.neighbour[index2] += 1
        agent2.neighbour[index1] += 1
        agent1.feeling[index2] = feeling
        agent2.feeling[index1] = feeling

        #doing the transaction
        agent1.money -= np.round(amount,3)
        agent2.money += np.round(amount,3)
        agent1.approval += np.round(amount*worth_ratio1 + agreement_point,3)
        agent2.approval -= np.round(amount*worth_ratio1 + agreement_point,3)
        
        if agent1.approval < 0.001:
            agent1.approval = 0
        if agent2.approval < 0.001:
            agent2.approval = 0
        
        agent1.worth_ratio = (amount*worth_ratio1 + agreement_point) / amount # = approval/money
        agent2.worth_ratio = (amount*worth_ratio1 + agreement_point) / amount # eqaul for both.

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

        agent1.neighbour_concatenation(index1,index2,t)
        agent2.neighbour_concatenation(index2,index1,t)
        
    return acceptance

# =============================================================================
def explore(index,t):
    '''choose another agent which is not in his memory
    uses proposition_2 (similar situation)
    before calling this function we have to check if the transaction was rewarding
    something have to be done in case there is no neighbours
    repetitious neighbours should be avoided'''
    
    agent = A[index]
    mask = np.ones(N,dtype=bool)
    mask[index] = False
    agent_active_neighbour = list(agent.active_neighbour.keys())
    self_similarity = agent.situation
    
    num_explore[t-1] += 1
    
    if len(agent_active_neighbour) != N-1:
        if len(agent_active_neighbour) != 0:
            chosen_neighbour_index = np.random.choice(agent_active_neighbour,p=list(agent.active_neighbour.values()))
            situation = A[chosen_neighbour_index].situation
            mask[agent_active_neighbour] = False
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
                    if len(other_agent.active_neighbour) != 0:
                        
                        nearest_choice = 1 #maximum possible situation difference
                        for k in other_agent.active_neighbour.keys():
                            diff_abs = np.abs(A[k].situation - self_similarity)
                            if diff_abs < nearest_choice:
                                nearest_choice = diff_abs
                                nearest_choice_index = k
                            
                        p = other_agent.active_neighbour[nearest_choice_index]
                        acceptance = np.random.choice([0,1],p=[1-p,p])
                        
#                        acceptance = 1
                        if acceptance == 1:
                            transaction(index,other_index,t)
                    else:
                        transaction(index,other_index,t)
                            
                    if other_index in agent.active_neighbour:  #which means transaction has been accepted
                        similarity_tracker[index].append(other_situation)
                        break
        else:
            other_index = np.random.choice(np.arange(N)[mask])
            other_agent = A[other_index]
            other_situation = other_agent.situation
            if len(other_agent.active_neighbour) != 0:

                nearest_choice = 1 #maximum possible situation difference
                for k in other_agent.active_neighbour.keys():
                    diff_abs = np.abs(A[k].situation - self_similarity)
                    if diff_abs < nearest_choice:
                        nearest_choice = diff_abs
                        nearest_choice_index = k
                    
                p = other_agent.active_neighbour[nearest_choice_index]
                acceptance = np.random.choice([0,1],p=[1-p,p])
                
#                acceptance = 1
                if acceptance == 1:
                    transaction(index,other_index,t)
            else:
                transaction(index,other_index,t)
            
            if other_index in agent.active_neighbour:
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
        os.mkdir(current_path+'\\runned_files'+version)
    except OSError:
        print ("version already exists")
        
    try:
        os.mkdir(current_path+'\\runned_files'+version+'\\N%d_T%d'%(N,T))
    except OSError:
        print ("Creation of the directory failed")
    
    path = 'runned_files'+version+'\\N%d_T%d\\'%(N,T)
#    np.save(path+'Agents.npy',A)
#    np.save(path+'Tracker.npy',tracker)
    with open(path + 'Agents.pkl','wb') as agent_file:
        pickle.dump(A,agent_file,pickle.HIGHEST_PROTOCOL)

    with open(path + 'Tracker.pkl','wb') as tracker_file:
        pickle.dump(tracker,tracker_file,pickle.HIGHEST_PROTOCOL)
        
    return
# =============================================================================
"""Parameters"""#XXX

N = 100
T = 10*N
similarity = 0.05                   #how much this should be?
memory_size = 10                    #contains the last memory_size number of transaction times
transaction_percentage = 0.3        #percent of amount of money the first agent proposes from his asset 
num_of_tries = 20                   #in function explore()
threshold_percentage = np.full(N,1) #the maximum amount which the agent is willing to give
normalization_factor = 1            #used in transaction(). what should be?
prob0_magnify_factor = 0.35         #this is in probability() for changing value so that it can take advantage of arctan
prob1_magnify_factor = 3
prob2_magnify_factor = 1
alpha = 1                           #in short-term effect of the frequency of transaction
beta = 0.3                          #in long-term effect of the frequency of transaction
param = 20                           #a normalizing factor in assigning the acceptance probability. It normalizes difference of money of both sides

"""Initial Condition"""

situation_arr = np.random.random(N) #randomly distributed
#money = np.full(N,2)        #may have distribution
#money = np.random.normal(loc=4,scale=1,size=N)
#money = 1 + situation_arr * 2
#money = np.zeros(N)
#money = np.random.random(N) * 2
money = np.round(situation_arr[:] * 9 + 1 ,decimals=3)
approval = np.full(N,5.5)     #may have distribution
#approval = 1 + situation_arr * 2
#approval = np.round(situation_arr[:] * 9 + 1 ,decimals=3)

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
        person_active_neighbour_size = len(person.active_neighbour)
        exploration_probability = (N-1-person_active_neighbour_size)/(N-1)#(2*N-2)
        explore_prob_array[t-1] += exploration_probability
        if person_active_neighbour_size != 0: #memory is not empty
            rand = np.random.choice([1,0],size=1,p=[1-exploration_probability,exploration_probability])
            if rand==1:
                
                person_active_neighbour = np.array(list(person.active_neighbour.keys()))
                if person_active_neighbour_size < num_of_tries:
                    num_of_choice = person_active_neighbour_size
                else:
                    num_of_choice = num_of_tries
                choice_arr = np.zeros(num_of_choice,dtype=int)
                for k in np.arange(num_of_choice):
                    choice_arr[k] , chosen_index = person.second_agent(i,person_active_neighbour)
                    person_active_neighbour = np.delete(person_active_neighbour, chosen_index)
                
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
    if t>2:
        tracker.get_list('worth_ratio',t-3)
    explore_prob_array[t-1] /= N


print(datetime.now() - start_time)
# =============================================================================
"""Write File"""
version = '\\4_param_2' #XXX
#save_it(version)

# =============================================================================
"""Analysis and Measurements"""
def plot_general(self,array,title=''):
    plt.figure()
    plt.plot(array)
    plt.title(title)
    return


analyse = Analysis_Tools_Homans.Analysis(N,T,memory_size,A)
analyse.draw_graph_weighted_colored()

a_money       = analyse.array('money')
a_approval    = analyse.array('approval')
a_worth_ratio = analyse.array('worth_ratio')
a_neighbour   = analyse.array('neighbour')
a_value       = analyse.array('value')
a_time        = analyse.array('time')
a_probability = analyse.array('probability')
a_utility = analyse.array('utility')

analyse.hist('money')
analyse.hist_log_log('money')
analyse.hist('approval')
#analyse.hist_log_log('approval')
#analyse.hist('degree')
#analyse.hist_log_log('degree')
#analyse.hist('value')
#analyse.hist_log_log('value')
#analyse.hist('probability')
#analyse.hist_log_log('probability')
#analyse.hist('utility')
#analyse.hist_log_log('utility')

#tracker.plot('self_value',title='Self Value')
tracker.plot('valuable_to_others',title='How Much Valuable to Others')
tracker.plot('worth_ratio',title='worth_ratio Evolution by Time')
#tracker.trans_time_visualizer(3,'Transaction Time Tracker')
#
tracker.plot_general(num_transaction_tot, title='Number of Transaction Vs. Time')
tracker.plot_general(num_explore, title='Number of Explorations Vs. Time')
#tracker.plot_general(agreement_tracker, title='agreement Point')
#tracker.plot_general(acceptance_tracker, title='Threshold Acceptance Over Time')
#
plt.figure()
plt.plot(p0_tracker[::2])
plt.plot(p1_tracker[::2])
plt.plot(p2_tracker[::2])
plt.title('P0 & P1 & P2')
#tracker.hist_general(p0_tracker,title='p0')
#tracker.hist_general(p1_tracker,title='p1')
#tracker.hist_general(p2_tracker,title='p2')
#tracker.hist_log_log_general(p0_tracker,title='P0')
#tracker.hist_log_log_general(p1_tracker,title='P1')
#tracker.hist_log_log_general(p2_tracker,title='P2')

plt.figure()
for i in np.arange(N):
    plt.plot(A[i].neighbour)
plt.title('A[i]s number of transaction with others')

#analyse.degree_vs_attr()

#plt.figure()
#for i in np.arange(N):
#    plt.plot(similarity_tracker[i])
#plt.ylim([0,1.1])
#plt.title('Similarity Tracker')

analyse.assortativity('money')
analyse.assortativity('approval')
analyse.assortativity('degree')
analyse.assortativity('situation')
analyse.assortativity('worth_ratio')

#tracker.hist_general(a_probability[a_probability>0.1])
#tracker.hist_log_log_general(a_probability[a_probability>0.1])
#array = a_value[a_value>0.8]
#plt.figure()
#plt.xscale('log')
#plt.yscale('log')
#bins=np.logspace(np.log10(np.amin(array)),np.log10(np.amax(array)),12)
#plt.hist(array,bins=bins)
#plt.title('Probability log-log Historgram bigger than 0.1')


"""Time Evaluation"""
duration = 500  # millisecond
freq = 2000  # Hz
winsound.Beep(freq, duration)
print (datetime.now() - start_time)
