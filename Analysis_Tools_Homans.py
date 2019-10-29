"""
Created on Fri Sep 13 12:53:00 2019
@author: Taha Enayat
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class Analysis:
    
    def __init__(self,number_agent,total_time,size,a_matrix,path,num_transaction,explore_prob_arr,*args,**kwargs):
        
        self.memory_size = size
        self.a_matrix = a_matrix
        self.N = number_agent
        self.total_time = total_time
        self.path = path
        
#        if string =='from_file': #load from file
#            current_path = os.getcwd()
#            path = '\\runned_files\\N%d_T%d_memory_size%d\\'%(self.N,self.total_time,self.memory_size)
#            
#            title = kwargs.get('file_title','Agents.pkl')
#            with open(current_path+path+title,'rb') as agent_file:
#                self.a_matrix = pickle.load(agent_file)
#                
#        if string =='in_memory': #file already ran and A[i]s are available
#            self.a_matrix = args[0]
        
        self.G = self._graph_construction('trans_number',num_transaction,explore_prob_arr)
        return
    
    def _graph_construction(self,graph_type,num_transaction,explore_prob_arr,**kwargs):
        G = nx.Graph()
        
        if graph_type == 'trans_number':
            sampling_time = kwargs.get('sampling_time',0)
            tracker = kwargs.get('tracker_obj',None)
            
            friendship_num = self.friendship_point(num_transaction,explore_prob_arr)
            if tracker != None:
                for i in np.arange(self.N):
                    for j in self.a_matrix[i].active_neighbor.keys():
                        trans_last_value = tracker.trans_time[sampling_time,i,j]
                        if True in (tracker.trans_time[sampling_time:,i,j] > (trans_last_value + friendship_num) ):
                            G.add_edge(i,j)
            else:                
                for i in np.arange(self.N):
                    for j in self.a_matrix[i].active_neighbor.keys():
                        if self.a_matrix[i].neighbor[j] > friendship_num:
                            G.add_edge(i,j)
                        
        node_attr_dict = { i:{'situation':0,'money':0,'worth_ratio':0,'others_feeling':0} for i in G.nodes() }
        for i in G.nodes():
            node_attr_dict[i]['situation'] = float(self.a_matrix[i].situation)
            node_attr_dict[i]['money'] = float(self.a_matrix[i].money)
            node_attr_dict[i]['worth_ratio'] = float(self.a_matrix[i].worth_ratio)
            node_attr_dict[i]['approval'] = float(self.a_matrix[i].approval)
            node_attr_dict[i]['others_feeling'] = float(self.array('others_feeling_for_agent')[i])
            node_attr_dict[i]['asset'] = float(self.a_matrix[i].asset)
        
        nx.set_node_attributes(G,node_attr_dict)
        return G
    
    
    def draw_graph_weighted_colored(self):
        plt.figure()
        print("Size of G is:", self.G.number_of_nodes())
#        edgewidth = [ d['weight'] for (u,v,d) in self.G.edges(data=True)]
        color = [ self.a_matrix[u].situation for u in self.G.nodes()]
        size = [self.a_matrix[u].asset*10 for u in self.G.nodes()]
#        pos = nx.spring_layout(self.G)
        pos = nx.kamada_kawai_layout(self.G)
        
#        nx.draw(self.G, pos=pos, with_labels = True, node_size=100, font_size=8, width=np.array(edgewidth), node_color=s)
        nx.draw(self.G, pos=pos, with_labels = True, node_size=size, font_size=8, node_color=color, width=0.2)
#        nx.draw(self.G, pos=pos, with_labels = True, node_size=150, font_size=8, width=np.array(edgewidth))
        plt.savefig(self.path+'graph')
        return

    def draw_graph(self):
        """
        it will draw the main graph
        """
        plt.figure()
        print("Size of G is:", self.G.number_of_nodes())
        pos_nodes = nx.spring_layout(self.G)
#        pos_nodes = nx.kamada_kawai_layout(self.G)
        
        nx.draw(self.G,pos = pos_nodes, with_labels = True, node_size=50, font_size=6, width=0.3)
        node_list = list(self.G.nodes())
        pos_attrs = {}
        for node, coords in pos_nodes.items():
            pos_attrs[node] = (coords[0], coords[1] + 0.04)
        node_attrs = { node:float("{0:.2f}".format(self.a_matrix[node].situation)) for node in node_list}
        custom_node_attrs = {}
        for node, attr in node_attrs.items():
            custom_node_attrs[node] = attr
        nx.draw_networkx_labels(self.G, pos_attrs, labels=custom_node_attrs,font_size=8)
        return

    def array(self,what_array):
        ref = {}
        
        if what_array == 'degree':
            ref[what_array] = [self.G.degree(n) for n in self.G.nodes()]
            return ref[what_array]
        
        if what_array == 'money':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].money
            return ref[what_array]

        if what_array == 'approval':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].approval
            return ref[what_array]

        if what_array == 'worth_ratio':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].worth_ratio
            return ref[what_array]
        
        if what_array == 'neighbor':
            ref[what_array] = np.zeros((self.N,self.N))
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].neighbor
            return ref[what_array]
        
        if what_array == 'value':
            ref[what_array] = np.zeros((self.N,self.N))
            for i in np.arange(self.N):
                for j in self.a_matrix[i].active_neighbor.keys():
                    if self.a_matrix[i].neighbor[j] < self.memory_size:
                        where = self.a_matrix[i].neighbor[j]-1 #last value in memory
                    else:
                        where = self.memory_size-1
                    ref[what_array][i,j] = self.a_matrix[i].value[j, where ]
            return ref[what_array]
            
        if what_array == 'time':
            ref[what_array] = np.zeros((self.N,self.N,self.memory_size))
            for i in np.arange(self.N):
                for j in np.arange(self.N):
                    if self.a_matrix[i].neighbor[j] != 0:
                        ref[what_array][i,j] = self.a_matrix[i].time[j]
                        #ref[what_array][i,j] = self.a_matrix[i].value[j]
                    else:
                        ref[what_array][i,j] = -1
            return ref[what_array]
        
        if what_array == 'probability':
            ref[what_array] = np.zeros((self.N,self.N))
            for i in np.arange(self.N):
                for j in self.a_matrix[i].active_neighbor.keys():
                    ref[what_array][i,j] = self.a_matrix[i].active_neighbor[j]
            return ref[what_array]

        if what_array == 'utility':
            ref[what_array] = np.zeros((self.N,self.N))
            for i in np.arange(self.N):
                for j in self.a_matrix[i].active_neighbor.keys():
                    ref[what_array][i,j] = self.a_matrix[i].active_neighbor[j] * self.a_matrix[j].active_neighbor[i]
            return ref[what_array]
        
        if what_array == 'others_feeling_for_agent':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array] += self.a_matrix[i].feeling[:]
            return ref[what_array]/np.sum(ref[what_array])
        
        if what_array == 'asset':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].asset
            return ref[what_array]
        
        if what_array == 'situation':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].situation
            return ref[what_array]
    
    def hist(self,what_hist):
        plt.figure()
        if what_hist == 'value' or what_hist == 'probability' or what_hist == 'utility':
            plt.hist(self.array(what_hist).flatten(),bins=50)
        else:
            plt.hist(self.array(what_hist),bins=15)
        title = what_hist+' histogram'
        plt.title(title)
        plt.savefig(self.path+title)
        return
    
    def hist_log_log(self,what_hist):
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        array = self.array(what_hist)
        if what_hist == 'value' or what_hist == 'probability' or what_hist == 'utility':
            bins=np.logspace(np.log10(np.amin(array.flatten()[array.flatten()>0])),np.log10(np.amax(array.flatten()[array.flatten()>0])),20)
            plt.hist(array.flatten(),bins=bins)
        else:
            bins=np.logspace(np.log10(np.amin(array)),np.log10(np.amax(array)),20)
            plt.hist(array,bins=bins)
        title = what_hist+' histogram log-log'
        plt.title(title)
        plt.savefig(self.path+title)
        return
    
    def topology_chars(self):

        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        G0 = self.G.subgraph(Gcc[0])
        H = nx.gnm_random_graph(self.G.number_of_nodes(),self.G.number_of_edges())
        Hcc = sorted(nx.connected_components(H), key=len, reverse=True)
        H0 = H.subgraph(Hcc[0])
        cc = nx.average_clustering(self.G)
        cc_r = nx.average_clustering(H)
        asph = nx.average_shortest_path_length(G0)
        asph_r = nx.average_shortest_path_length(H0)
        
        title = 'Topological Charateristics.txt'
        topol_file = open(self.path+title,'w')
        topol_file.write('Size of the Giant Component is: '+str(G0.number_of_nodes())+' with '+str(G0.number_of_edges())+' edges'+'\n')
        topol_file.write('Average Shortert Path Length'+'\n')
        topol_file.write(str(asph)+'\n')
        topol_file.write('Clustering Coeficient'+'\n')
        topol_file.write(str(cc)+'\n')
        topol_file.write('Small-Worldness'+'\n')
        topol_file.write(str( (cc/cc_r) / (asph/asph_r) )+'\n')
        topol_file.write('\n')
        topol_file.write('The Corresponding Random Graph Has:'+'\n')
        topol_file.write('Shortert Path Length: '+str(asph_r)+'\n')
        topol_file.write('Clustering Coeficient: '+str(cc_r)+'\n')
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
        return
    
    def num_of_transactions(self):
        plt.figure()
        for i in np.arange(self.N):
            plt.plot(self.a_matrix[i].neighbor)
        title = 'A[i]s number of transaction with others'
        plt.title(title)
        plt.savefig(self.path+title)
        return

    def assortativity(self):
        title = 'Assortativity.txt'
        assort_file = open(self.path + title,'w')
        for attr in self.G.nodes(data=True)[0].keys():
            assort_file.write('assortativity according to '+attr+' is: '+'\n')
            assort_file.write(str(nx.attribute_assortativity_coefficient(self.G,attr))+'\n'+'\n')
        return 
    
    def money_vs_situation(self,path):
        plt.figure()
        plt.scatter(self.array('situation'),self.array('money'))
        title = 'Money Vs Situation'
        plt.title(title)
        plt.savefig(self.path+title)
        return
    
    def transaction_vs_asset(self):
        transaction = np.zeros(self.N)
        asset = self.array('asset')
        for i in np.arange(self.N):
            transaction[i] = np.sum(self.a_matrix[i].neighbor)
        bins = 15
        x = np.linspace(np.min(asset),np.max(asset),num=bins+1,endpoint=True)
        width = x[1] - x[0]
        y = np.zeros(bins)
        for bin_index in np.arange(bins):
            for i in np.arange(self.N):
                if asset[i] < x[bin_index+1] and x[bin_index] < asset[i]:
                    y[bin_index] += transaction[i]
        plt.figure()
        plt.bar(x[:-1] + width/2,y,width=width)
        title = 'Transaction Vs Asset'
        plt.title(title)
        plt.savefig(self.path+title)
        return
    
    def rich_club(self,normalized=False):
        rich_club_coef = nx.rich_club_coefficient(self.G,normalized=normalized)
        print(rich_club_coef)
        rc_array = np.zeros(len(rich_club_coef))
        for i in np.arange(len(rich_club_coef)):
            rc_array[i] = rich_club_coef[i]
        plt.figure()
        plt.plot(rc_array)
        if normalized:
            title = 'Rich Club Normalized'
        else:
            title = 'Rich Club NonNormalized'
        plt.title(title)
        plt.savefig(self.path+title)
        return
    
    def friendship_point(self,num_transaction,explore_prob_arr):
        """ When we consider someone as friend
        Or in other words: how many transactions one agent with an agent means that they are friends
        """
        alpha = num_transaction[:] / ((1 - explore_prob_arr[:]) * self.N)
        T_eff = np.sum(alpha)
        beta = 1
        friendship_num = int(np.ceil(beta * T_eff / self.N))
        print('friendship point:',friendship_num)
        print('average transaction',np.average(num_transaction))
        return friendship_num

    

class Tracker:
    
    def __init__(self,number_agent,total_time,size,a_matrix):
        
        self.a_matrix = a_matrix
        self.total_time = total_time
        self.memory_size = size
        self.N = number_agent
        
        """Trackers"""
        global self_value,valuable_to_others,worth_ratio,exploration,exploration_avg
        self_value = np.zeros((self.total_time,self.N))
        valuable_to_others = np.zeros((self.total_time,self.N))
        worth_ratio = np.zeros((self.total_time-2,self.N))
        self.trans_time = np.ones((self.total_time,self.N,self.N))
        self.correlation_mon = np.zeros(self.total_time)
        self.correlation_situ = np.zeros(self.total_time)
        
    def update_A(self,a_matrix):
        self.a_matrix = a_matrix
        return
    
    def get_path(self,path):
        self.path = path
        return
        
    def get_list(self,get_list,t):
        
        if get_list == 'self_value':
            self_value[t] = np.sum(self._array('value'),axis = 1)
        if get_list == 'valuable_to_others':
            valuable_to_others[t] = np.sum(self._array('value'),axis = 0)
        if get_list == 'worth_ratio':
            worth_ratio[t] = self._array('worth_ratio')
        if get_list == 'trans_time':
            for i in np.arange(self.N):
                self.trans_time[t,i,:] = np.copy(self.a_matrix[i].neighbor)
        if get_list == 'correlation_mon':
            self.correlation_mon[t] = self.correlation_money_situation()
        if get_list == 'correlation_situ':
            self.correlation_situ[t] = self.correlation_situation_situation()
        
    def make_dynamic_trans_time_graph(self,graph):
        """
        adds the start time attribute to the graph
        """
        edge_attr_dict = {(x,y):{'start':self.total_time,'end':self.total_time} for x,y in graph.edges()}
        
        for x,y in graph.edges():
            edge_attr_dict[(x,y)]['start'] = self._edge_start_time(x,y)
            edge_attr_dict[(x,y)]['end'] = self.total_time
            
        nx.set_edge_attributes(graph,edge_attr_dict)
        return graph
    
    def _edge_start_time(self,x,y):
        """
        compute the start time for an edge
        """
        time = np.where(self.trans_time[:,x,y] >= 5)[0][0]
        return int(time)
    
    def _array(self,what_array):
        ref = {}
        
        if what_array == 'value':
            ref[what_array] = np.zeros((self.N,self.N))
            for i in np.arange(self.N):
                for j in np.arange(self.N):
                    if self.a_matrix[i].neighbor[j] != 0:
                        if self.a_matrix[i].neighbor[j] < self.memory_size:
                            where = self.a_matrix[i].neighbor[j]-1 #last value in memory
                        else:
                            where = self.memory_size-1
                        ref[what_array][i,j] = self.a_matrix[i].value[j, where ]
                    else:
                        ref[what_array][i,j] = 0
            return ref[what_array]
                    
        if what_array == 'worth_ratio':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].worth_ratio
            return ref[what_array]
        
    
    def plot(self,what_array,**kwargs):
        ref = {'self_value': self_value,
               'valuable_to_others': valuable_to_others,
               'worth_ratio': worth_ratio,
               'correlation_mon': self.correlation_mon,
               'correlation_situ': self.correlation_situ}
        plt.figure()
        title = kwargs.get('title',what_array)
        plt.title(title)
        plt.plot(ref[what_array])
        plt.savefig(self.path+title)
        return
    
    def plot_general(self,array,title=''):
        plt.figure()
        plt.plot(array)
        plt.title(title)
        plt.savefig(self.path+title)
        return
    
    def hist_general(self,array,title=''):
        plt.figure()
        plt.hist(array)
        plt.title(title)
        plt.savefig(self.path+title)
        return
    
    def hist_log_log_general(self,array,title=''):
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        bins=np.logspace(np.log10(np.amin(array)),np.log10(np.amax(array)),20)
        plt.hist(array,bins=bins)
        plt.title(title+' histogram log-log'+' N={} T={}'.format(self.N,self.total_time))
        plt.savefig(self.path+title+' histogram log-log'+' N={} T={}'.format(self.N,self.total_time))
        return
    
    def index_in_arr(array,value):
        return np.where( array == value )[0][0]
    
    def trans_time_visualizer(self,agent_to_watch,title):
        """
        it will show each node transaction transcript.
        """
        fig, ax = plt.subplots(nrows=1,ncols=1)
        im = ax.imshow(self.trans_time[:,agent_to_watch,:].astype(float),aspect='auto')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('number of transactions', rotation=-90, va="bottom")
        plt.title(title)
        plt.savefig(self.path+title)
        return
    
    def return_arr(self):
        return self.trans_time
    
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
    