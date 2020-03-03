"""
Created on Fri Sep 13 12:53:00 2019
@author: Taha Enayat, Mohsen Mehrani
"""
import sys
import os
pd = {'win32':'\\', 'linux':'/'}
if sys.platform.startswith('win32'):
    plat = 'win32'
elif sys.platform.startswith('linux'):
    plat = 'linux'
sys.path.insert(1, os.getcwd() + pd[plat]+'homans_tools')

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from graph_tools_glossary import Graph_related_tools,Community_related_tools
from agents_properties_tools import arrays_glossary


class properties_alteration(arrays_glossary):
    def property_evolution(self,property_id):
        ref = {'money':self.agents_money,
               'approval':self.agents_approval,
               'asset':self.agents_asset}
        property_arr = ref[property_id]
        
        fig1, (ax1,ax2) = plt.subplots(nrows=2,ncols=1)
        fig2, (ax3,ax4) = plt.subplots(nrows=2,ncols=1)
        
        ax1.title.set_text('last&first ' + property_id + ' vs situation')
        ax1.scatter(self.array('situation'),property_arr[0,:],c='r')
        
        for t in np.arange(1,self.T,self.T-1,dtype = int):
            ax1.scatter(self.array('situation'),property_arr[t,:])
        
        ax2.title.set_text(property_id + ' growth vs situation')
        ax2.scatter(self.array('situation'),property_arr[self.T-1,:] - property_arr[0,:])
        
        ax3.title.set_text('initial vs last'+property_id)
        ax3.scatter(property_arr[0,:],property_arr[self.T-1,:])
        ax4.title.set_text(property_id+' growth')
        ax4.scatter(property_arr[0,:],property_arr[self.T-1,:] - property_arr[0,:])
        
        fig1.savefig(self.path + 'P ' + property_id + ' growth vs situation')
        fig2.savefig(self.path + 'P ' + 'initial vs last'+property_id)
        return
    
    def correlation_money_situation(self):
        money = np.zeros(self.N)
        situation = np.zeros(self.N)
        for i in np.arange(self.N):
            money[i] = self.a_matrix[i].money
            situation[i] = self.a_matrix[i].situation
        money_avg = np.average(money)
        situation_avg = np.average(situation)
        correlation = np.sum( (money-money_avg)*(situation-situation_avg) ) / np.sqrt(np.sum( (money-money_avg)**2 ) * np.sum( (situation-situation_avg)**2 ) )
        return correlation
    
    def correlation_situation_situation(self):
        situation = np.zeros(self.N)
        situation_neighbor = np.zeros(self.N)
        for i in np.arange(self.N):
            situation[i] = self.a_matrix[i].situation
            length = len(self.a_matrix[i].active_neighbor)
            if length != 0:
                for j in self.a_matrix[i].active_neighbor.keys():
                    situation_neighbor[i] += self.a_matrix[j].situation
                situation_neighbor[i] /= length
        avg_situation = np.average(situation)
        avg_situation_n = np.average(situation_neighbor)
        numerator = np.sum( (situation-avg_situation)*(situation_neighbor-avg_situation_n))
        denominator = np.sqrt(np.sum( (situation-avg_situation)**2 ) * np.sum( (situation_neighbor-avg_situation_n)**2 ) )
        correlation = numerator / denominator
        return correlation
    
    def correlation_growth_situation(self,survey_property_id,base_property_id):
        
        survey_ref = {'money':self.agents_money,
                      'approval':self.agents_approval,
                      'asset':self.agents_asset}
        base_ref = { base_property_id:self.array(base_property_id)}
        
        for key in survey_ref.keys():
            base_ref['initial_'+ key] = survey_ref[key][0,:]
        
        fig, ax = plt.subplots(nrows=1,ncols=1)
        
        survey_arr = survey_ref[survey_property_id]
        base_arr = base_ref[base_property_id]
        corr = np.zeros(self.T)
        for t in np.arange(self.T):
            corr[t] = np.corrcoef(survey_arr[t,:]-survey_arr[0,:],base_arr[:])[0,1]
        plt.plot(np.arange(self.T),corr)
        ax.set_title('correlation between '+survey_property_id+' growth'+' & '+base_property_id)
        fig.savefig(self.path + 'correlation between '+survey_property_id+' growth'+' & '+base_property_id)
        return
    def correlation_pairplots(self,**kwargs):
        """
        we need to know how do correlations among attributes changes over time.
        """
        all_nodes = kwargs.get('all_nodes',False)
        if all_nodes == True:
            present_nodes = range( self.N )
            status = 'all nodes'
        elif all_nodes == False:
            present_nodes = kwargs.get('present_nodes',None)    
            status = 'graph nodes'
            
        nodes_mask = [ node in present_nodes for node in range(self.N)]
        attributes = {0:self.agents_money,1:self.agents_asset,2:self.agents_approval}
        attributes_name = {0:'money',1:'asset',2:'approval'}
        fig,axes = plt.subplots(3,3)
        
        for i in range(len(attributes)):
            for j in range(len(attributes)):
                axes[i,j].plot( [ np.corrcoef(attributes[i][x,nodes_mask],attributes[j][x,nodes_mask])[0,1] for x in range(self.T) ])
                if j == 0:
                    axes[i,j].set_ylabel( attributes_name[i] )
                if i == len(attributes_name) - 1:
                    axes[i,j].set_xlabel( attributes_name[j] )
                axes[i,j].set_ylim([-1.05,1.05])
                    
#        plt.ylim(-1.05,1.05)
        fig.savefig(self.path + 'correlations pair plots versus time for '+status)
        plt.close()
        
        return
    
    pass

class hist_plot_tools():
    def plot(self,what_array,**kwargs):
        ref = {'self_value': self.self_value,
               'valuable_to_others': self.valuable_to_others,
               'worth_ratio': self.worth_ratio,
               'correlation_mon': self.correlation_mon,
               'correlation_situ': self.correlation_situ}
        plt.figure()
        title = kwargs.get('title',what_array)
        alpha = kwargs.get('alpha',1)
        plt.title(title)
        plt.plot(ref[what_array],alpha=alpha)
        plt.savefig(self.path+title)
        plt.close()
        return
    
    def plot_general(self,array,title='',**kwargs):
        plt.figure()
        plt.plot(array)
        explore = kwargs.get('explore',False)
        if explore:
            N = kwargs.get('N',100)
            plt.ylim(0,1.05*N)
        plt.title(title)
        plt.savefig(self.path+title)
        plt.close()
        return
    
    def hist_general(self,array,title=''):
        plt.figure()
        plt.hist(array)
        plt.title(title)
        plt.savefig(self.path+title)
        plt.close()
        return
    
    def hist_log_log_general(self,array,title=''):
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        bins=np.logspace(np.log10(np.amin(array)),np.log10(np.amax(array)),20)
        plt.hist(array,bins=bins)
        plt.title(title+' Histogram log-log'+' N={} T={}'.format(self.N,self.T))
        plt.savefig(self.path+title+' Histogram Log-Log')
        plt.close()
        return
    pass

class Analysis(Graph_related_tools,properties_alteration): #XXX
    def __init__(self,number_agent,total_time,size,a_matrix,path,*args,**kwargs):
        
        self.memory_size = size
        self.a_matrix = a_matrix
        self.N = number_agent
        self.T = total_time
        self.path = path
        return

    def hist(self,what_hist):
        plt.figure()
        dic = {'value','probability','utility','neighbor'}
        array = self.array(what_hist)
        if what_hist in dic:
            array = array.flatten()[array.flatten()>0]
            plt.hist(array,bins='auto')
        else:
            plt.hist(array,bins='auto')
        title = 'Histrogram of ' + what_hist
        plt.title(title)
        plt.savefig(self.path+title)
        plt.close()
        return
    
    def hist_log_log(self,what_hist,semilog=False):
        plt.figure()
        plt.xscale('log')
        title = 'Histogram log-log of ' + what_hist
        if semilog == False:
            plt.yscale('log')
        else:
            title = 'Histogram semi-log of ' + what_hist
        array = self.array(what_hist)
        if what_hist in ['value','probability','utility','neighbor']:
            array = array.flatten()[array.flatten()>0]
            bins=np.logspace(np.log10(np.amin(array)),np.log10(np.amax(array)),15)
            plt.hist(array.flatten(),bins=bins)
        else:
            bins=np.logspace(np.log10(np.amin(array)),np.log10(np.amax(array)),15)
            plt.hist(array,bins=bins)
        plt.title(title)
        plt.savefig(self.path+title)
        plt.close()
        return


    def agents_prob_sum(self):
        a_prob = self.array('probability')
        agents_self_value = np.sum(a_prob,axis = 0)
        a_asset = self.array('asset')
        stacked_array = np.transpose(np.stack((agents_self_value,a_asset)))
        
        stacked_array_sorted = stacked_array[np.argsort(stacked_array[:,0])]
        
        dic=dict(zip(agents_self_value,np.arange(self.N)))
        label=np.zeros(self.N,dtype=int)
        for i,x in enumerate(stacked_array_sorted[:,0]):
            label[i] = dic[x]
        
        plt.figure()
        title = 'probability to be chosen by other agents'
        plt.title(title)
        
        plt.scatter(np.arange(self.N),stacked_array_sorted[:,0],c = stacked_array_sorted[:,1] )
        
        for x,y in zip(np.arange(self.N),stacked_array_sorted[:,0]):
            plt.text(x-0.1,y+0.2,str(label[x]),fontsize=8)
        plt.savefig(self.path+title)
        plt.close()
        return

    def degree_vs_attr(self):
        G_deg = dict(self.G.degree)
        deg_attr = [ [self.a_matrix[x].situation,G_deg[x]] for x in G_deg.keys() ]
        deg_attr = sorted(deg_attr, key=lambda a_entry: a_entry[0])
        deg_attr = np.transpose(deg_attr)
        plt.figure()
        plt.xlabel('attractiveness')
        plt.ylabel('degree')
        title = 'How famous are the most attractive agents?'
        plt.title(title)
        plt.scatter(deg_attr[0],deg_attr[1])
        plt.savefig(self.path+title)
        plt.close()
        return
    
    def num_of_transactions(self):
        plt.figure()
        for i in np.arange(self.N):
            plt.plot(self.a_matrix[i].neighbor,alpha=(1-i/self.N*2/3))
        title = 'Number of Transaction {0:.3g}'.format(self.transaction_average)
        plt.title(title)
        plt.savefig(self.path+title+'.png')
        plt.close()
        return 
    
    def money_vs_situation(self,path):
        plt.figure()
        plt.scatter(self.array('situation'),self.array('money'))
        title = 'Money Vs Situation'
        plt.title(title)
        plt.savefig(self.path+title)
        plt.close()
        return
    
    def transaction_vs_property(self,what_prop):
        transaction = np.zeros(self.N)
        array = self.array(what_prop)
        for i in np.arange(self.N):
            transaction[i] = np.sum(self.a_matrix[i].neighbor)
        bins = 15
        x = np.linspace(np.min(array),np.max(array),num=bins+1,endpoint=True)
        width = x[1] - x[0]
        y = np.zeros(bins)
        for bin_index in np.arange(bins):
            counter = 0
            for i in np.arange(self.N):
                if array[i] < x[bin_index+1] and x[bin_index] < array[i]:
                    y[bin_index] += transaction[i]
                    counter += 1
            if counter != 0:
                y[bin_index] /= counter #normalization
        plt.figure()
        plt.bar(x[:-1] + width/2,y,width=width)
        title = 'Transaction vs ' + what_prop
        plt.title(title)
        plt.savefig(self.path+title)
        plt.close()
        return

    def plot_general(self,path,array,title='',second_array=None,indicator=True):
        plt.figure()
        if indicator:
            plt.plot(array)
        else:
            for i in np.arange(len(array)):
                plt.plot(array[i])
        if second_array != None:
            plt.plot(second_array)
        plt.title(title)
        plt.savefig(path + title)
        plt.close()
        return

    
class Tracker(properties_alteration,hist_plot_tools): #XXX
    
    def __init__(self,number_agent,total_time,size,a_matrix,trans_saving_time_interval,saving_time_step,boolean=False,*args,**kwargs):
        
        self.a_matrix = a_matrix
        self.T = total_time
        self.memory_size = size
        self.N = number_agent
#        self.sampling_time = 2000
#        if self.sampling_time > self.T:
#            self.sampling_time = self.T
#        self.boolean = boolean
#        self.boolean, self.saving_time_step = kwargs.get('to_save_last_trans',[False,None])
#        self.saving_time_step = kwargs.get('saving_time_step',self.saving_time_step)
        self.saving_time_step = saving_time_step
        self.trans_saving_time_interval = trans_saving_time_interval
        
        """Trackers"""
        self.self_value = np.zeros((self.T,self.N))
        self.valuable_to_others = np.zeros((self.T,self.N))
        self.worth_ratio = np.zeros((self.T-2,self.N))
        
#        if self.boolean:
        self.trans_time = np.zeros((self.trans_saving_time_interval,self.N,self.N))
        
        self.sample_time_trans = np.zeros((self.N,self.N))
        self.correlation_mon = np.zeros(self.T)
        self.correlation_situ = np.zeros(self.T)
        self.agents_money  = np.zeros((self.T,self.N))
        self.agents_asset  = np.zeros((self.T,self.N))
        self.agents_approval  = np.zeros((self.T,self.N))
        
        self.rejection_time = np.zeros((self.T,16))
#        self.rejection_time = np.zeros((self.N,16))
        
    def update_A(self,a_matrix):
        self.a_matrix = a_matrix
        return
    
    def get_path(self,path):
        self.path = path
        return
        
    def get_list(self,get_list,t,array=None):
        
        if get_list == 'self_value':
            self.self_value[t] = np.sum(self.array('value'),axis = 1)
        if get_list == 'valuable_to_others':
            self.valuable_to_others[t] = np.sum(self.array('value'),axis = 0)
        if get_list == 'worth_ratio':
            self.worth_ratio[t] = self.array('worth_ratio')
            
        if get_list == 'money':
            self.agents_money[t] = self.array('money')
        if get_list == 'asset':
            self.agents_asset[t] = self.array('asset')
        if get_list == 'approval':
            self.agents_approval[t] = self.array('approval')
        
        if get_list == 'sample_time_trans':
            for i in np.arange(self.N):
                self.sample_time_trans[i,:] = np.copy(self.a_matrix[i].neighbor)
                
        if get_list == 'trans_time':
            for i in np.arange(self.N):
                self.trans_time[t,i,:] = np.copy(self.a_matrix[i].neighbor)
                
        if get_list == 'correlation_mon':
            self.correlation_mon[t] = self.correlation_money_situation()
        if get_list == 'correlation_situ':
            self.correlation_situ[t] = self.correlation_situation_situation()
            
        if get_list == 'rejection':
            self.rejection_time = np.copy(array)
        
    def make_dynamic_trans_time_graph(self,graph):
        """
        adds the start time attribute to the graph
        """
        edge_attr_dict = {(x,y):{'start':self.T,'end':self.T} for x,y in graph.edges()}
        
        for x,y in graph.edges():
            edge_attr_dict[(x,y)]['start'] = self._edge_start_time(x,y)
            edge_attr_dict[(x,y)]['end'] = self.T
            
        nx.set_edge_attributes(graph,edge_attr_dict)
        return graph
    
    def _edge_start_time(self,x,y):
        """
        compute the start time for an edge
        """
        time = np.where(self.trans_time[:,x,y] >= self.friendship_num)[0][0]
        return int(time)
    
    def index_in_arr(array,value):
        return np.where( array == value )[0][0]
    
    def trans_time_visualizer(self,agent_to_watch,title,**kwargs):
        """
        it will show each node transaction transcript.
        """
#        sort_by = kwargs.get('sorting_feature','situation')
        
        fig, ax = plt.subplots(nrows=1,ncols=1)
        
#        sort_arr = self.array(sort_by)
##        sort_arr_sorted = np.sort(sort_arr)
##        x_label_list = ['%.2f'%(sort_arr_sorted[i]) for i in range(self.N) ]
##        ax.set_xticklabels(x_label_list)
        
#        im = ax.imshow(self.trans_time[:,agent_to_watch,np.argsort(sort_arr)].astype(float),aspect='auto')
        agent_trans_time = self.trans_time[:,agent_to_watch,:].astype(float)
        show_arr = np.zeros((self.trans_saving_time_interval-1,self.N))
        for t in np.arange(self.trans_saving_time_interval-1):
            show_arr[t] = agent_trans_time[t+1,:] - agent_trans_time[t,:]
#        print('trans time',np.size(agent_trans_time[:,0]))
#        print('show arr',np.size(show_arr[:,0]))
        im = ax.imshow(show_arr,aspect='auto')

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('number of transactions', rotation=-90, va="bottom")
        plt.title(title+', agent {0} with asset={1:.3g} & situation={2:.2f}'.format(agent_to_watch,self.a_matrix[agent_to_watch].asset,self.a_matrix[agent_to_watch].situation))
        plt.savefig(self.path + title + ' agent {0}'.format(agent_to_watch))
        plt.close()
        return
    
    def rejection_history(self):
        binary = [0,1]
        conditions_glossary = [(x,y,z,w) for x in binary for y in binary for z in binary for w in binary]
        conditions_glossary_dict = { cond:x for cond,x in zip(conditions_glossary,range(16))}
        conditions_glossary_string = ['{0}'.format(x) for x in conditions_glossary]
        
        total_rejection_cases = np.sum(self.rejection_time,axis = 0)
        plt.figure()
        plt.bar(conditions_glossary_string,total_rejection_cases)
        
        for i,v in enumerate(total_rejection_cases):
            plt.text( x = i , y = total_rejection_cases[i], s = str(int(total_rejection_cases[i])) , rotation = -90)
        plt.xlabel('(acceptance_worth , acceptance_thr , acceptance_asset , acceptance_util)')
        title = 'rejection history of all running time'
        plt.title(title)
        plt.savefig(self.path + title)
        plt.close()
        return

    def valuability(self):
        fig, ax = plt.subplots(nrows=1,ncols=1)
        asset = self.array('asset')
#        asset_sort = np.sort(asset)
#        x_label_list = np.array(['{0:.2f}'.format(asset_sort[int(self.N/5)*i]) for i in np.arange(5) ])
#        x_label_list = np.concatenate(([0],x_label_list))
#        ax.set_xticklabels(x_label_list)
        valuable_to_others_normalized = np.zeros((self.T,self.N))
        for i in np.arange(self.N):
            valuable_to_others_normalized[:,i] = self.valuable_to_others[:,i] / len(self.a_matrix[i].active_neighbor)
        im = ax.imshow(valuable_to_others_normalized[:,np.argsort(asset)].astype(float),aspect='auto')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('value sum', rotation=-90, va="bottom")
        title = 'How Much Valuable to Others (sorted based on asset & normalized)'
        plt.title(title)
        plt.savefig(self.path + title)
        plt.close()
        return
    



    

    
    
