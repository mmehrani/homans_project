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
from decimal import Decimal

class properties_alteration(arrays_glossary):
    
    def property_evolution(self,property_id):
        """ 
        Compares initial and finial amount of given property
        """
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
        """
        Calculates correlation of money and situation in a given time
        (it is used to plot this correlation throughout time)
        """
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
        """ 
        Calculates the correlation of situation of agents and their neighbors.
        For dimensions to be equal, it replaces average sitaution of agent's neighbors.
        """
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
        """ 
        Claculates the correlation of the growth of one property and another property.
        """
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
        We need to know how do correlations among attributes changes over time.
        """
        nodes_selection = kwargs.get('nodes_selection','all_nodes')
        if nodes_selection == 'all_nodes':
            present_nodes = range(self.N)
            foldername = ''
            status = 'all nodes'
            
        elif nodes_selection == 'graph_nodes':
            foldername = ''
            status = 'graph nodes'
            present_nodes = kwargs.get('present_nodes',None)
            
        elif nodes_selection.split('_')[0] == 'community': #community_nodes_# is the form
            foldername = 'communities_attr_correlation'
            present_nodes = kwargs.get('present_nodes',None)
            status = 'community #{}'.format(nodes_selection.split('_')[2])
            try:
                os.mkdir(self.path + foldername)
            except:
                pass
        nodes_mask = [ node in present_nodes for node in range(self.N)]
        attributes = {0:self.agents_money,1:self.agents_asset,2:self.agents_approval,3:np.concatenate((self.worth_ratio[:2,:],self.worth_ratio), axis = 0)}
        attributes_name = {0:'money',1:'asset',2:'approval',3:'worth_ratio'}
        fig,axes = plt.subplots(nrows = 4,ncols = 4 , figsize = (16,9))
        
        for i in range(len(attributes)):
            for j in range(len(attributes)):
                axes[i,j].plot( [ np.corrcoef(attributes[i][x,nodes_mask],attributes[j][x,nodes_mask])[0,1] for x in range(self.T) ])
                if j == 0:
                    axes[i,j].set_ylabel( attributes_name[i] )
                if i == len(attributes_name) - 1:
                    axes[i,j].set_xlabel( attributes_name[j] )
                axes[i,j].set_ylim([-1.05,1.05])
                    
        fig.savefig( os.path.join( self.path,foldername,'correlations pair plots versus time for '+status))
        plt.close()
        return
    
    def property_variation(self):
        """ 
        Creates a figure indicating standard deviation of properties,
        and a figure indicating the normalized version of standard deviation (devided to average)
        
        Each color is a different community and the dashed line the avarage of 
        the standard deviation of a property in the whole agents.
        """
        nei_approval_var,nei_money_var,nei_asset_var,nei_worth_var = np.zeros(self.N),np.zeros(self.N),np.zeros(self.N),np.zeros(self.N)
        mean_approval,mean_money,mean_asset,mean_worth = np.zeros(self.N),np.zeros(self.N),np.zeros(self.N),np.zeros(self.N)
        
        communities_parts = self.modularity_communities #XXX
        # communities_parts = self.modularity_communitiesx
        
        for agent in np.arange(self.N):
            nei_approval = [self.a_matrix[j].approval for j in self.a_matrix[agent].active_neighbor]
            nei_money = [self.a_matrix[j].money for j in self.a_matrix[agent].active_neighbor]
            nei_asset = [self.a_matrix[j].asset for j in self.a_matrix[agent].active_neighbor]
            nei_worth = [self.a_matrix[j].worth_ratio for j in self.a_matrix[agent].active_neighbor]
            
            nei_approval_var[agent] = np.var(nei_approval)
            nei_money_var[agent] = np.var(nei_money)
            nei_asset_var[agent] = np.var(nei_asset)
            nei_worth_var[agent] = np.var(nei_worth)
            
            mean_approval[agent] = np.mean(nei_approval)
            mean_money[agent] = np.mean(nei_money)
            mean_asset[agent] = np.mean(nei_asset)
            mean_worth[agent] = np.mean(nei_worth)
        
        nei_approval_var = np.sqrt(nei_approval_var[~np.isnan(nei_approval_var)])
        nei_money_var = np.sqrt(nei_money_var[~np.isnan(nei_money_var)])
        nei_asset_var = np.sqrt(nei_asset_var[~np.isnan(nei_asset_var)])
        nei_worth_var = np.sqrt(nei_worth_var[~np.isnan(nei_worth_var)])
        
        mean_approval = mean_approval[~np.isnan(mean_approval)]
        mean_money= mean_money[~np.isnan(mean_money)]
        mean_asset= mean_asset[~np.isnan(mean_asset)]
        mean_worth= mean_worth[~np.isnan(mean_worth)]
        
        mean_approval_var = np.mean(nei_approval_var)
        mean_money_var = np.mean(nei_money_var)
        mean_asset_var = np.mean(nei_asset_var)
        mean_worth_var = np.mean(nei_worth_var)
        
        mean_rel_approval = np.mean(nei_approval_var/mean_approval)
        mean_rel_money = np.mean(nei_money_var/mean_money)
        mean_rel_asset = np.mean(nei_asset_var/mean_asset)
        mean_rel_worth = np.mean(nei_worth_var/mean_worth)
        
        plt.figure()
        splitter = 0
        for com in communities_parts:
            plt.plot(range(splitter,splitter+len(com)),nei_approval_var[com])
            plt.plot(range(splitter,splitter+len(com)),nei_money_var[com])
            plt.plot(range(splitter,splitter+len(com)),nei_asset_var[com])
            plt.plot(range(splitter,splitter+len(com)),nei_worth_var[com])
            splitter += len(com)
        
        plt.plot([0,self.N],[mean_approval_var,mean_approval_var], ls = 'dashed',label = 'approval')
        plt.plot([0,self.N],[mean_money_var,mean_money_var], ls = 'dashed',label = 'money')
        plt.plot([0,self.N],[mean_asset_var,mean_asset_var], ls = 'dashed',label = 'asset')
        plt.plot([0,self.N],[mean_worth_var,mean_worth_var], ls = 'dashed',label = 'worth ratio')
        plt.title("Standard Deviation of Agent's Neighbor Properties")
        plt.legend()
        plt.savefig(self.path + 'SD '+'mean approval={0:.3g} money={1:.3g} asset={2:.3g} WR={3:.3g}.png'.format(mean_approval_var,mean_asset_var,mean_money_var,mean_worth_var))
        plt.close()
        
        plt.figure()
        splitter = 0
        for com in communities_parts:
            plt.plot(range(splitter,splitter+len(com)),nei_approval_var[com]/mean_approval[com])
            plt.plot(range(splitter,splitter+len(com)),nei_money_var[com]/mean_money[com])
            plt.plot(range(splitter,splitter+len(com)),nei_asset_var[com]/mean_asset[com])
            plt.plot(range(splitter,splitter+len(com)),nei_worth_var[com]/mean_worth[com])
            splitter += len(com)

        plt.plot([0,self.N],[mean_rel_approval,mean_rel_approval], ls = 'dashed',label = 'approval')
        plt.plot([0,self.N],[mean_rel_money,mean_rel_money], ls = 'dashed',label = 'money')
        plt.plot([0,self.N],[mean_rel_asset,mean_rel_asset], ls = 'dashed',label = 'asset')
        plt.plot([0,self.N],[mean_rel_worth,mean_rel_worth], ls = 'dashed',label = 'worth ratio')
        plt.legend()
        plt.title("Relative Standard Deviation of Neighbors Properties")
        plt.savefig(self.path + 'SDR '+'mean approval={0:.3g} money={1:.3g} asset={2:.3g} WR={3:.3g}.png'.format(mean_rel_approval,mean_rel_money,mean_rel_asset,mean_rel_worth))
        plt.close()
        return
    
    def prob_nei_correlation(self):
        """ 
        Creates plot of correlation of probability of an agent and how many times that
        agent transacts with his neighbors.
        """        
        prob = self.array('probability')
        neighbor = self.array('neighbor')
        correlation = np.zeros(self.N)
        for i in np.arange(self.N):
            correlation[i] = np.corrcoef(prob[i],neighbor[i])[0,1]
        plt.figure()
        plt.ylim([-1.05,1.05])
        plt.plot(sorted(correlation))
        title = 'Correlation of Probability and Number of Transaction (Neighbor)'
        plt.title(title)
        plt.savefig(self.path + 'correlation of probability and neighbor')
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
        
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(array)
        
        ax.set_xlabel('Time steps')
        
        explore = kwargs.get('explore',False)
        if explore:
            ax.set_ylabel('Number of Explorations')
            deg_value = np.polyfit(np.arange(self.T - 1000, self.T),np.log(array[-1000:]),1)
            ax.plot( np.arange(self.T) ,np.exp( deg_value[1] + deg_value[0]*np.arange(self.T) ) ,
                    color = 'k',ls = 'dashed',alpha = 0.3, label = '{:.2e} t + {:.2e}'.format(Decimal(deg_value[0]),Decimal(deg_value[1])))
            ax.set_yscale('log')
            plt.legend()
            ax.set_title(title)
            
        trans = kwargs.get('trans',False)
        if trans:
            ax.set_ylabel('Number of Transactions')
            avg = np.average(array)
            ax.set_title(title + ' average={:.2f}'.format(avg))
        
        fig.savefig(self.path + title)
        plt.close()
        return
    
    def hist_general(self,array,title=''):
        plt.figure()
        plt.hist(array)
        plt.title(title)
        plt.savefig(self.path + title)
        plt.close()
        return
    
    def hist_log_log_general(self,array,title=''):
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        array = np.array(array)
        array = array[array > 0]
        bins=np.logspace(np.log10(np.amin(array)),np.log10(np.amax(array)),20)
        plt.hist(array,bins=bins)
        plt.title(title + ' Histogram log-log' + ' N={} T={}'.format(self.N,self.T))
        plt.savefig(self.path + title + ' Histogram Log-Log')
        plt.close()
        return
    
    pass


class Analysis(Graph_related_tools,properties_alteration): #XXX
    """
    Main class for analysis (mostly not time dependent ones)
    It uses inheritance to communicate with other classes
    """
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
        plt.savefig(self.path + title)
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
        """ 
        Creates a figure in which neighbor array of each agent is shown
        """
        plt.figure()
        for i in np.arange(self.N):
            plt.plot(self.a_matrix[i].neighbor,alpha=(1-i/self.N*2/3))
        title = 'Number of Transaction {0:.3g}'.format(self.transaction_average)
        plt.title(title)
        plt.savefig(self.path+title+'.png')
        plt.close()
        return 
    
    def money_vs_situation(self,path):
        """ 
        Creates a figure which horizontal axis is situation and vertical axis is money and
        each point in the figure is one agent with corresponding money and situation
        """
        plt.figure()
        plt.scatter(self.array('situation'),self.array('money'))
        title = 'Money Vs Situation'
        plt.title(title)
        plt.savefig(self.path+title)
        plt.close()
        return
    
    def transaction_vs_property(self,what_prop):
        """ 
        How many transactions happened in a specific property (like money)
        => Histogram of transaction according to a property
        """
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

    def plot_general(self,path,array,title='',second_array=None,indicator=True,**kwargs):
        """ 
        Used in graph_related function
        """
        plt.figure()
        if indicator:
            plt.plot(array)
        else:
            label = kwargs.get('label',None)
            if label != None:
                for i in np.arange(len(array)):
                    plt.plot(array[i],label=label[i])
                plt.legend()
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
    """ 
    A class for analyse through time
    """    
    def __init__(self,number_agent,total_time,size,a_matrix,trans_saving_time_interval,saving_time_step,boolean=False,*args,**kwargs):
        
        self.a_matrix = a_matrix
        self.T = total_time
        self.memory_size = size
        self.N = number_agent
        self.saving_time_step = saving_time_step
        self.trans_saving_time_interval = trans_saving_time_interval
        
        """Trackers"""
        self.self_value = np.zeros((self.T,self.N))
        self.valuable_to_others = np.zeros((self.T,self.N))
        self.worth_ratio = np.zeros((self.T-2,self.N))
        self.trans_time = np.zeros((self.trans_saving_time_interval,self.N,self.N))
        self.sample_time_trans = np.zeros((self.N,self.N))
        self.correlation_mon = np.zeros(self.T)
        self.correlation_situ = np.zeros(self.T)
        self.agents_money  = np.zeros((self.T,self.N))
        self.agents_asset  = np.zeros((self.T,self.N))
        self.agents_approval  = np.zeros((self.T,self.N))
        self.rejection_time = np.zeros((self.T,16))
        return
        
    def update_A(self,a_matrix):
        self.a_matrix = a_matrix
        return
    
    def get_path(self,path):
        self.path = path
        return
        
    def get_list(self,get_list,t,array=None):
        """ 
        Get and update lists from Homans.py main
        """
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
    
    def trans_time_visualizer(self,agent_to_watch,title,**kwargs):
        """
        it will show the trace of transactions of one agent through time
        Horizontal axis is other agents, and vertica axis is time.
        """
        fig, ax = plt.subplots(nrows=1,ncols=1)
        agent_trans_time = self.trans_time[:,agent_to_watch,:].astype(float)
        show_arr = np.zeros((self.trans_saving_time_interval-1,self.N))
        for t in np.arange(self.trans_saving_time_interval-1):
            show_arr[t] = agent_trans_time[t+1,:] - agent_trans_time[t,:]
        im = ax.imshow(show_arr,aspect='auto')

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('number of transactions', rotation=-90, va="bottom")
        plt.title(title+', agent {0} with asset={1:.3g} & situation={2:.2f}'.format(agent_to_watch,self.a_matrix[agent_to_watch].asset,self.a_matrix[agent_to_watch].situation))
        plt.savefig(self.path + title + ' agent {0}'.format(agent_to_watch))
        plt.close()
        return
    
    def rejection_history(self):
        """ 
        Tracks rejections and acceptances in transaction function
        """
        binary = [0,1]
        conditions_glossary = [(x,y,z,w) for x in binary for y in binary for z in binary for w in binary]
        conditions_glossary_dict = { cond:x for cond,x in zip(conditions_glossary,range(16))}
        conditions_glossary_string = ['{0}'.format(x) for x in conditions_glossary]

        total_rejection_cases = np.sum(self.rejection_time,axis = 0)
        plt.figure(figsize=(16, 9))
        plt.bar(conditions_glossary_string,total_rejection_cases)

        for i,v in enumerate(total_rejection_cases):
            plt.text( x = i , y = total_rejection_cases[i], s = str(int(total_rejection_cases[i])) , rotation = -90)
        plt.xlabel('(acceptance_worth , acceptance_thr , acceptance_asset , acceptance_util)')
        title = 'rejection history of all running time'
        plt.title(title)
        plt.savefig(self.path + title)
        plt.close()
        
        plt.figure(figsize=(16, 9))
        total_rejection_at_moment = np.sum(self.rejection_time,axis = 1)
        for i,v in enumerate(total_rejection_cases):
            plt.plot(self.rejection_time[:,i]/total_rejection_at_moment,label = conditions_glossary_string[i] ,alpha = 0.5)
        plt.legend()
        plt.xlabel('(acceptance_worth , acceptance_thr , acceptance_asset , acceptance_util)')
        title = 'rejection history versus time in relative'
        plt.title(title)
        plt.savefig(self.path + title)
        plt.close()
        return

    def valuability(self):
        fig, ax = plt.subplots(nrows=1,ncols=1)
        asset = self.array('asset')
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
