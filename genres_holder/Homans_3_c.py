"""
Created on Mon Aug 12 10:12:03 2019
@author: Taha Enayat, Mohsen Mehrani

Main file
Model's engine
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#import winsound
import pickle
import Analysis_Tools_Homans
import os
import sys
from decimal import *
"""Platform Detection"""
pd = {'win32':'\\', 'linux':'/'}
if sys.platform.startswith('win32'):
    plat = 'win32'
elif sys.platform.startswith('linux'):
    plat = 'linux'
start_time = datetime.now()

# =============================================================================
"""Class"""
class NegativeProbability(Exception):
    pass

class Agent():
    """
    Properties and variables related to an agent
    """
    def __init__(self,money,approval,situation):
        self.money = money
        self.approval = approval
        self.neighbor = np.zeros(N,dtype=int) #number of interactions
        self.value = np.full(N,-1,dtype=float)
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
    
    def worth_ratio_calculator(self):
        """ 
        calculates worth ratio i.e. averages over neighbors' money and approval. 
        Used in transaction function.
        """
        self.n_avg = {'money':0, 'approval':0}
        for j in self.active_neighbor.keys():
            self.n_avg['money'] += A[j].money
            self.n_avg['approval'] += A[j].approval

        self.n_avg['money'] += self.money
        self.n_avg['approval'] += self.approval
        self.n_avg['money'] = self.n_avg['money'] / (len(self.active_neighbor)+1)
        self.n_avg['approval'] = self.n_avg['approval'] / (len(self.active_neighbor)+1)
        
        self.n_average = self.n_avg['approval'] / self.n_avg['money']
        return self.n_average

    def probability_factor(self,neighbor,t):
        '''
        Calculates the factor for choosing each neighbor that converts to probability in 
        neighbor_concatenation function.
        
        This factor is multiplication of effect of value (proposition3), frequency (proposition4),
        and feeling (proposition5).
        '''
        p0 = np.exp(self.value[neighbor] * prob0_magnify_factor)
        p1 = self.frequency_to_probability(neighbor,t) * prob1_magnify_factor - (prob1_magnify_factor -1)
        p2 = np.exp(self.feeling[neighbor]) * prob2_magnify_factor - (prob2_magnify_factor -1)

        # p0 = 1.0      #for when we need to turn off the effect
        # p1 = 1.0      #for when we need to turn off the effect
        # p2 = 1.0      #for when we need to turn off the effect
        
        p0_tracker.append(p0)   #tracking
        p1_tracker.append(p1)
        p2_tracker.append(p2)
        
        probability = p0 * p1 * p2      #not normalized. normalization occurs in neighbor_concatenation()
        return Decimal(probability).quantize(Decimal('1e-5'),rounding = ROUND_DOWN) if probability < 10**8 else Decimal(10**8)
    
    def frequency_to_probability(self,neighbor,t):
        """
        Homans' proposition 4. 
        
        Although he doesn't talk about effect of longterm memory on probability,
        it is here to see whether it makes the results more real or not.
        """
        mask = (self.time[neighbor] > t-10) & (self.time[neighbor] != -1)
        n1 = np.size(self.time[neighbor][mask])
        short_term = 1 - alpha * (n1/10)
        
        # n2 = self.neighbor[neighbor]
        # long_term = 1 + beta * (n2 * len(self.active_neighbor) /(t*np.average(num_transaction_tot[:t-1]) ) ) 
        long_term = 1.0     #for when we need to turn off the effect
        
        prob = short_term * long_term
        return prob
    
    
    def neighbor_concatenation(self,self_index,new_neighbor,t):
        """ 
        Adds new neighbor to memory and converts factor obtained from probability_factor()
        function to probability (that sums to one).
        """
        for j in self.active_neighbor.keys():
            self.active_neighbor[j] *= self.sigma
            
        grade_new_neighbor = self.probability_factor(new_neighbor,t)
        if new_neighbor in self.active_neighbor:
            self.sigma += grade_new_neighbor - self.active_neighbor[new_neighbor]
        else:
            self.sigma += grade_new_neighbor
            
        self.active_neighbor[new_neighbor] = grade_new_neighbor
        for j in self.active_neighbor.keys():
            if j!=new_neighbor:
                self.active_neighbor[j] /= self.sigma
                self.active_neighbor[j] = Decimal( str(self.active_neighbor[j]) ).quantize(Decimal('1e-5'),rounding = ROUND_DOWN)
                
        if new_neighbor in self.active_neighbor:
            self.active_neighbor[new_neighbor] = 1 - ( sum(self.active_neighbor.values()) -  self.active_neighbor[new_neighbor])
        else:
            self.active_neighbor[new_neighbor] = 1 -  sum(self.active_neighbor.values()) 
            
        """Error finding"""
        if self.active_neighbor[new_neighbor] < 0:
            raise NegativeProbability('self index:',self_index,'neighbor',new_neighbor)
        elif np.size(np.array(list(self.active_neighbor.values()))[np.array(list(self.active_neighbor.values()))>1]) != 0:
            raise NegativeProbability('self index:',self_index,'neighbor',new_neighbor)
        elif sum(list(self.active_neighbor.values())) > 1.01 or sum(list(self.active_neighbor.values())) < 0.99:
            raise NegativeProbability('not one',sum(list(self.active_neighbor.values())))

        return

    def second_agent(self,self_index,self_active_neighbor):
        """
        Homans' proposition 6
        
        Returns an agent in memory with maximum utility to intract with.
        Utility = Value * Acceptance Probability
        """
        """Proposition 6"""
        i = 0
        Max = 0
        for j in self_active_neighbor:
            value = self.value[j]
            other_probability = float(A[j].active_neighbor[self_index])
            utility = value * other_probability
            if utility >= Max:
                Max = utility
                chosen_agent = j
                chosen_agent_index = i
            i += 1
            
        """random choice"""
        # chosen_agent_index = np.random.choice(range(len(self_active_neighbor)))
        # chosen_agent = self_active_neighbor[chosen_agent_index]

        return chosen_agent , chosen_agent_index
    
# =============================================================================
"""Functions"""
def transaction(index1,index2,t):
    """ 
    Transaction with two agents
    agent1 proposes to agent2
    Uses proposition 3 (value) and proposition 5 (feeling)
    """
    agent1 = A[index1]
    agent2 = A[index2]
    number_of_transaction1 = agent1.neighbor[index2]
    number_of_transaction2 = agent2.neighbor[index1]
    
    if len(agent1.active_neighbor) != 0:
        worth_ratio1 = agent1.worth_ratio_calculator()
    else:
        worth_ratio1 = agent1.worth_ratio
    if len(agent2.active_neighbor) != 0:
        worth_ratio2 = agent2.worth_ratio_calculator()
    else:
        worth_ratio2 = agent2.worth_ratio
        
    amount = transaction_percentage * agent1.money
    agreement_point = (worth_ratio2 - worth_ratio1)/(worth_ratio2 + worth_ratio1) * amount * worth_ratio1 #x=(E2-E1/E2+E1)*AE1

    """Acceptances"""
    
    """although it seems obvious that the agent2 has to accept the transaction according to 
    what he thinks of agent1, here in the code it is redundancy;
    Because in the code we are sure that agent1 have chosen agent2 according to maximizing
    utility, i.e. agent2 is "the chosen one"!
    
    The problem if this acceptance is on is that probabilities attributed to neighbors are
    in the order of 1/N and with N=100 it means that most of the time transactions are rejected.
    """
    # if index1 in agent2.active_neighbor:
    #     p = agent2.active_neighbor[index1]
    #     acceptance_util = np.random.choice([0,1],p=[1-p,p])
    # else:
    #     acceptance_util = 1
    acceptance_util = 1     #for turning off the effect of utility acceptance
    
    if agent2.approval > 0.001 and agent2.approval - ( np.round(amount*worth_ratio1 + agreement_point,3) ) > 0.001:
        acceptance_neg = 1      #not negative checking acceptance
    else: acceptance_neg = 0
        
    # if True:      #for turning off the effect of worth ratio acceptance
    if worth_ratio2 >= worth_ratio1:
        acceptance_worth = 1
    else:
        p = np.exp( -(worth_ratio1 - worth_ratio2)/normalization_factor )
        acceptance_worth = np.random.choice([0,1],p=[1-p,p])
    acceptance_worth = acceptance_worth * acceptance_neg
        
    p = np.exp( -np.abs(agent1.asset - agent2.asset)/param )
    acceptance_asset = np.random.choice([0,1],p=[1-p,p])
    # acceptance_asset = 1      #for turning off the effect of asset acceptance
        
    threshold = threshold_percentage[index2] * agent2.approval
    if threshold > (amount * worth_ratio1 + agreement_point):
        acceptance_thr = 1
    else: acceptance_thr = 0
    acceptance = acceptance_worth * acceptance_thr * acceptance_asset * acceptance_util
        
    acceptance_manager([acceptance_worth, acceptance_thr, acceptance_asset, acceptance_util],index1,t)  #tracking
    
    if acceptance:   #transaction accepts
        num_transaction_tot[t-1] += 1
        
        """Calculate feeling and value"""
        feeling = agreement_point / worth_ratio1    #is equal for both (from definition)
        value1 = + amount + agreement_point/worth_ratio1
        value2 = + amount

        agent1.neighbor[index2] += 1
        agent2.neighbor[index1] += 1
        agent1.feeling[index2] = feeling
        agent2.feeling[index1] = feeling

        """Updating memory"""
        agent1.money -= np.round(amount,3)
        agent2.money += np.round(amount,3)
        agent1.approval += np.round(amount*worth_ratio1 + agreement_point,3)
        agent2.approval -= np.round(amount*worth_ratio1 + agreement_point,3)
        agent1.worth_ratio = lamda * agent1.worth_ratio + (1-lamda) * (amount*worth_ratio1 + agreement_point) / amount
        agent2.worth_ratio = lamda * agent2.worth_ratio + (1-lamda) * (amount*worth_ratio1 + agreement_point) / amount
        agent1.asset_updater()
        agent2.asset_updater()
        agent1.value[index2] = value1
        agent2.value[index1] = value2
        
        asset_tracker[index1].append(agent1.asset)      #tracker
        asset_tracker[index2].append(agent2.asset)      #tracker
        
        if number_of_transaction1 < memory_size:        #if memory is not full
            empty_memory = number_of_transaction1
            agent1.time [index2,empty_memory] = t
        else:
            shift_memory( agent1 , index2)
            agent1.time [index2,memory_size-1] = t
            
        if number_of_transaction2 < memory_size:        #if memory is not full
            empty_memory = number_of_transaction2
            agent2.time [index1,empty_memory] = t
        else:
            shift_memory(agent2,index1)
            agent2.time [index1,memory_size-1] = t

        agent1.neighbor_concatenation(index1,index2,t)
        agent2.neighbor_concatenation(index2,index1,t)
        
    return acceptance

# =============================================================================
def shift_memory(agent,index):
    temp = np.delete(agent.time[index],0)
    agent.time[index] = np.concatenate((temp,[-1]))
    return

# =============================================================================
def acceptance_manager(accept_list,agent,t):
    """ 
    To track acceptances through time
    """
    dic_value = conditions_glossary_dict[tuple(accept_list)]
    rejection_agent[agent,dic_value] += 1
    rejection_time[t-1,dic_value] += 1
    return

# =============================================================================
def explore(index,t):
    """
    Chooses another agent which is not in his memory
    Uses proposition 2 (similar situation)
    """    
    agent = A[index]
    mask = np.ones(N,dtype=bool)
    mask[index] = False
    agent_active_neighbor = list(agent.active_neighbor.keys())
    self_similarity = agent.situation
    
    num_explore[t-1] += 1
    global counter_accept_nei, counter_accept_ran
    
    if len(agent_active_neighbor) != N-1:
        if len(agent_active_neighbor) != 0:
            
            """Finding neighbors of neighbors"""
            neighbors_of_neighbors_not_flat = []
            for j in agent_active_neighbor:
                neighbors_of_neighbors_not_flat.append(A[j].active_neighbor.keys())
            neighbors_of_neighbors = []
            for sublist in neighbors_of_neighbors_not_flat:
                for item in sublist:
                    neighbors_of_neighbors.append(item)
            neighbors_of_neighbors = list(set(neighbors_of_neighbors))
            neighbors_of_neighbors.remove(index)
            for nei in neighbors_of_neighbors:
                if nei in agent_active_neighbor:
                    neighbors_of_neighbors.remove(nei)

            """Proposing"""
            if len(neighbors_of_neighbors) != 0:
                model_neighbor_index = np.random.choice(agent_active_neighbor,p=list(agent.active_neighbor.values())) #Bias neighbor
                model_situation = A[model_neighbor_index].situation
                
                if len(neighbors_of_neighbors) >= num_of_tries2:
                    arri_choice = np.random.choice(neighbors_of_neighbors,size=num_of_tries2,replace=False)
                else:
                    arri_choice = np.array(neighbors_of_neighbors)
                
                for other_index in arri_choice:
                    other_situation = A[other_index].situation
                    
                    if other_situation > (model_situation-similarity) and other_situation < (model_situation+similarity): #if matches the criteria
                        
                        """Waiting for the answer of the proposed neighbor"""
                        other_agent = A[other_index]
                        if len(other_agent.active_neighbor) != 0:
                            nearest_choice = 1      #maximum possible situation difference
                            for k in other_agent.active_neighbor.keys():
                                diff_abs = np.abs(A[k].situation - self_similarity)
                                if diff_abs < nearest_choice:
                                    nearest_choice = diff_abs
                                    nearest_choice_index = k
                            p = other_agent.active_neighbor[nearest_choice_index]
                            acceptance = np.random.choice([0,1],p=[1-p,p])
    
                            if acceptance == 1:
                                transaction(index,other_index,t)
                        else:
                            transaction(index,other_index,t)
                                
                        if other_index in agent.active_neighbor:  #which means transaction has been accepted
                            counter_accept_nei += 1
                            break

                anyof = True
                for i in arri_choice:
                    if i in agent.active_neighbor:
                        anyof = False
                        
                """When nobody is the right fit, the agent looks for a random agent"""
                if anyof:
                    mask[agent_active_neighbor] = False
                    if np.size(mask[mask==True]) >= num_of_tries3:
                        arri_choice = np.random.choice(np.arange(N)[mask],size=num_of_tries3,replace=False)     #difference with above
                    else:
                        num_true_in_mask = np.size(mask[mask==True])
                        arri_choice = np.random.choice(np.arange(N)[mask],size=num_true_in_mask,replace=False)
                    
                    for other_index in arri_choice:
                        other_situation = A[other_index].situation
                        if other_situation > (model_situation-similarity) and other_situation < (model_situation+similarity):
                            other_agent = A[other_index]
                            if len(other_agent.active_neighbor) != 0:
                                nearest_choice = 1      #maximum possible situation difference
                                for k in other_agent.active_neighbor.keys():
                                    diff_abs = np.abs(A[k].situation - self_similarity)
                                    if diff_abs < nearest_choice:
                                        nearest_choice = diff_abs
                                        nearest_choice_index = k
                                p = other_agent.active_neighbor[nearest_choice_index]
                                acceptance = np.random.choice([0,1],p=[1-p,p])
                                if acceptance == 1:
                                    transaction(index,other_index,t)
                            else:
                                transaction(index,other_index,t)
                            if other_index in agent.active_neighbor:  #which means transaction has been accepted
                                counter_accept_ran += 1
                                break

            else:
                """Nobody is in memory so choose with no model neighbor"""
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
                    if acceptance == 1:
                        transaction(index,other_index,t)
                else:
                    transaction(index,other_index,t)
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
                if acceptance == 1:
                    transaction(index,other_index,t)
            else:
                transaction(index,other_index,t)

    return

# =============================================================================
def make_directories(version):
    """ 
    Making directories before running the simulation
    It also makes a file of initial conditions and parameters
    """
    current_path = os.getcwd()
    try: os.mkdir(current_path+pd[plat]+'runned_files')
    except OSError:
        print ("runned_files already exists")
    try: os.mkdir(current_path+pd[plat]+'runned_files'+pd[plat]+'N%d_T%d'%(N,T))
    except OSError:
        print ("version already exists")
    
    path = current_path+pd[plat]+'runned_files'+pd[plat]+'N%d_T%d'%(N,T)+pd[plat]+version+pd[plat]
    try: os.mkdir(path)
    except OSError:
        print ("Creation of the directory failed")
        
    with open(path + 'Initials.txt','w') as initf:
        initf.write(str(N)+'\n')
        initf.write(str(T)+'\n')
        initf.write(str(sampling_time)+'\n')
        initf.write(str(saving_time_step)+'\n')
        initf.write(str(version)+'\n')
        initf.write('respectively: \n')
        initf.write('N, T, sampling time, saving time step, version \n\n')
        
        initf.write(str(initial_for_trans_time) + ': initial for trans time \n')
        initf.write(str(trans_saving_interval) + ': trans time interval \n')
        initf.write(str(similarity) + ': similarity \n')
        initf.write(str(num_of_tries1) + ': num of tries 1 \n')
        initf.write(str(num_of_tries2) + ': num of tries 2 \n')
        initf.write(str(num_of_tries3) + ': num of tries 3 \n')
        initf.write(str(prob0_magnify_factor) + ': probability 0 magnify factor \n')
        initf.write(str(prob1_magnify_factor) + ': probability 1 magnify factor \n')
        initf.write(str(prob2_magnify_factor) + ': probability 2 magnify factor \n')
        initf.write(str(lamda) + ': lambda \n')
        
    return path

def save_it(version,t):
    """ 
    Saves essential data and makes corresponding directories
    """
    global tracker
    current_path = os.getcwd()
    path = current_path+pd[plat]+'runned_files'+pd[plat]+'N%d_T%d'%(N,T)+pd[plat]+version+ pd[plat]+'0_%d'%(t)+pd[plat]
    try: os.mkdir(path)
    except OSError:
        print ("Creation of the subdirectory failed")
    
    with open(path + 'Agents.pkl','wb') as agent_file:
        pickle.dump(A,agent_file,pickle.HIGHEST_PROTOCOL)
        
    with open(path + 'Other_data.pkl','wb') as data:
        pickle.dump(num_transaction_tot[t-sampling_time:t],data,pickle.HIGHEST_PROTOCOL) #should save the midway num_trans
        pickle.dump(explore_prob_array,data,pickle.HIGHEST_PROTOCOL)
        pickle.dump(rejection_agent,data,pickle.HIGHEST_PROTOCOL)
        
    with open(path + 'Tracker.pkl','wb') as tracker_file:
        pickle.dump(tracker,tracker_file,pickle.HIGHEST_PROTOCOL)
        
    return path

# =============================================================================
"""Distinctive parameters"""        #necessary for recalling for analysis
N = 100                             #Number of agents

"""Parameters"""#XXX
similarity = 0.05                   #difference allowed between model neighbor and new found agent. in explore()
memory_size = 10                    #how many time of transaction for each agent is stored in memory of one agent
transaction_percentage = 0.1        #percent of amount of money the first agent proposes from his asset 
num_of_tries1 = 20                  #in main part
num_of_tries2 = 20                  #in function explore(); tries from neighbors of neighbors
num_of_tries3 = 1                   #in function explore(); tries from random agents (if no neighbor of neighbor have found)
threshold_percentage =np.full(N,1)  #the maximum amount which the agent is willing to give
normalization_factor = 1            #rejection rate of acceptance_worth; used in transaction()
prob0_magnify_factor = 0.5          #magnifying factor of P0; in probability_factor()
prob1_magnify_factor = 1            #magnifying factor of P1; in probability_factor(); use with caution
prob2_magnify_factor = 1            #magnifying factor of P2; in probability_factor(); use with caution
alpha = 1                           #in short-term effect of the frequency of transaction
beta = 3                            #in long-term effect of the frequency of transaction
param = 2                           #a normalizing factor in assigning the acceptance probability. It normalizes difference of money of both sides
lamda = 0                           #how much one agent relies on his last worth_ratio and how much relies on current transaction's worth_ratio
sampling_time = 1000                #time interval used for making network: [T-sampling_time , T]
saving_time_step = T                #for saving multiple files change it from T to your desired interval (better for T to be devidable to your number)
initial_for_trans_time = T - 1000   #initial time for trans_time to start recording 
trans_saving_interval = 1000        #the interval the trans_time will record

if sampling_time > T:
    sampling_time = T
if saving_time_step < sampling_time:
    saving_time_step = sampling_time

"""Initial Condition"""

situation_arr = np.random.random(N) #randomly distributed

#money = np.full(N,5.5)
money = np.round(np.random.rand(N) * 9 + 1 ,decimals=3)     #randomly between [1,10]

approval = np.full(N,5.5)
#approval = np.round(np.random.rand(N) * 9 + 1 ,decimals=3) #randomly between [1,10]

A = np.zeros(N,dtype=object)
for i in np.arange(N):
    A[i]=Agent( money[i], approval[i], situation_arr[i]) 

"""trackers"""
explore_prob_array = np.zeros(T)
num_transaction_tot = np.zeros(T)

rejection_time = np.zeros((T,16))
rejection_agent = np.zeros((N,16))
binary = [0,1]
conditions_glossary = [(x,y,z,w) for x in binary for y in binary for z in binary for w in binary]
conditions_glossary_dict = { cond:x for cond,x in zip(conditions_glossary,range(16))}
conditions_glossary_string = ['{0}'.format(x) for x in conditions_glossary]

tracker = Analysis_Tools_Homans.Tracker(N,T,memory_size,A,trans_saving_interval,saving_time_step)
num_explore = np.zeros(T)
p0_tracker = []
p1_tracker = []
p2_tracker = []
asset_tracker = [ [] for _ in np.arange(N) ]

counter_entrance = 0
counter_accept_nei = 0
counter_accept_ran = 0

"""preparing for writing files"""
path = make_directories(version)

# =============================================================================
"""Main"""

"""
Choose one agent, find another agent through calculating probability,
explores for new agent (expand memory)
"""
for t in np.arange(T)+1:#t goes from 1 to T
    """computations"""
    print(t,'/',T)
    tau = (t-1)
    shuffled_agents=np.arange(N)
    np.random.shuffle(shuffled_agents)
    
    for i in shuffled_agents:
        person = A[i]
        person_active_neighbor_size = len(person.active_neighbor)
        exploration_probability = (N-1-person_active_neighbor_size)/(N-1)#(2*N-2)
        explore_prob_array[tau] += exploration_probability
        if person_active_neighbor_size != 0:    #memory is not empty
            rand = np.random.choice([1,0],size=1,p=[1-exploration_probability,exploration_probability])
            
            if rand==1:
                person_active_neighbor = np.array(list(person.active_neighbor.keys()))
                if person_active_neighbor_size < num_of_tries1:
                    num_of_choice = person_active_neighbor_size
                else:
                    num_of_choice = num_of_tries1
                choice_arr = np.zeros(num_of_choice,dtype=int)
                for k in np.arange(num_of_choice):
                    choice_arr[k] , chosen_index = person.second_agent(i,person_active_neighbor)
                    person_active_neighbor = np.delete(person_active_neighbor, chosen_index)
                
                for j in choice_arr:
                    if transaction(i,j,t):
                        break
            else:
                counter_entrance += 1
                explore(i,t)
        else:
            counter_entrance += 1
            explore(i,t)
    
    """trackers"""
    tracker.update_A(A)
    tracker.get_list('self_value',tau)
    tracker.get_list('valuable_to_others',tau)
    tracker.get_list('correlation_mon',tau)
    tracker.get_list('correlation_situ',tau)
    tracker.get_list('money',tau)
    tracker.get_list('approval',tau)
    tracker.get_list('asset',tau)
    if t>2:
        tracker.get_list('worth_ratio',tau-2)
    if tau == saving_time_step - sampling_time:
        tracker.get_list('sample_time_trans',tau)
    if t % saving_time_step == 0 or t == 1:
        boolean = False
    if t % saving_time_step == 0 and t >= saving_time_step:
        tracker.get_list('rejection',tau,array=rejection_time)
        save_it(version,t) #Write File
    if t >= initial_for_trans_time and t < initial_for_trans_time + trans_saving_interval:
        boolean = True
    else:
        boolean = False
    t_prime = t - initial_for_trans_time
    if boolean:
        tracker.get_list('trans_time',t_prime)
    explore_prob_array[tau] /= N

# =============================================================================

"""Pre-Analysis and Measurements"""
tracker.get_path(path)
tracker.plot_general(explore_prob_array * N,title='Average Exploration Probability',explore=True,N=N)
tracker.plot_general(num_transaction_tot,title='Number of Transaction',trans=True)

plt.figure()
plt.plot(p0_tracker[::2])
plt.plot(p1_tracker[::2])
plt.plot(p2_tracker[::2])
plt.title('P0 & P1 & P2')
plt.savefig(path+'P0 & P1 & P2')
plt.close()
tracker.hist_general(p0_tracker,title='p0')
tracker.hist_general(p1_tracker,title='p1')
tracker.hist_general(p2_tracker,title='p2')
tracker.hist_log_log_general(p0_tracker,title='P0')
tracker.hist_log_log_general(p1_tracker,title='P1')
tracker.hist_log_log_general(p2_tracker,title='P2')

plt.figure()
for i in np.arange(N):
    plt.plot(asset_tracker[i])
plt.title('Asset Tracker')
plt.savefig(path+'Asset Tracker')
plt.close()

with open(path + 'Explore_data.txt','w') as ex_file:
    ex_file.write('Enterance to exploration \n')
    ex_file.write(str(counter_entrance) + '\n\n')
    ex_file.write('Total accepted explorations \n')
    ex_file.write(str(counter_accept_nei + counter_accept_ran) + '\n\n')
    ex_file.write('Accepted in neighbor of neighbor part \n')
    ex_file.write(str(counter_accept_nei) + '\n\n')
    ex_file.write('Accepted in random part \n')
    ex_file.write(str(counter_accept_ran) + '\n\n')
    ex_file.write('Neighbor to random ratio \n')
    ex_file.write(str(counter_accept_ran / counter_accept_nei) + '\n\n')
    ex_file.write('Total accepted to entrance ratio \n')
    ex_file.write(str((counter_accept_nei+counter_accept_ran) / counter_entrance) + '\n\n')
    ex_file.write('\nRun Time:')
    ex_file.write(str(datetime.now() - start_time))
    
"""Time Evaluation"""
duration = 500  # millisecond
freq = 2000  # Hz
#winsound.Beep(freq, duration)
print (datetime.now() - start_time)
