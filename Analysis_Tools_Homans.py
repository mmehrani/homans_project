"""
Created on Fri Sep 13 12:53:00 2019
@author: Taha Enayat, Mohsen Mehrani
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import community
import matplotlib.animation as animation
from networkx.algorithms import community as communityx


class Analysis: #XXX
    
    def __init__(self,number_agent,total_time,size,a_matrix,path,*args,**kwargs):
        
        self.memory_size = size
        self.a_matrix = a_matrix
        self.N = number_agent
        self.T = total_time
        self.path = path
        
#        if string =='from_file': #load from file
#            current_path = os.getcwd()
#            path = '\\runned_files\\N%d_T%d_memory_size%d\\'%(self.N,self.T,self.memory_size)
#            
#            title = kwargs.get('file_title','Agents.pkl')
#            with open(current_path+path+title,'rb') as agent_file:
#                self.a_matrix = pickle.load(agent_file)
#                
#        if string =='in_memory': #file already ran and A[i]s are available
#            self.a_matrix = args[0]
        
#        self.G = self.graph_construction('trans_number',num_transaction)
        return
    
    def graph_construction(self,graph_type,num_transaction,boolean=True,**kwargs):
        G = nx.Graph()
        if graph_type == 'trans_number':
#            sampling_time = kwargs.get('sampling_time',0)
            if self.T >= 1000:
                sampling_time = 1000
            else:
                sampling_time = int(self.T / 2)
                
            trans_time = kwargs.get('trans_time',None)
#            tracker = Tracker(self.N,self.T,self.memory_size,self.a_matrix)
            if boolean:
#                self.friendship_point(num_transaction,sampling_time)
                self.friendship_point(num_transaction)
            else:
                self.friendship_num = kwargs.get('fpoint')
            if trans_time != None:
                for i in np.arange(self.N):
                    for j in self.a_matrix[i].active_neighbor.keys():
#                        trans_last_value = tracker.trans_time[sampling_time,i,j]
                        trans_last_value = trans_time[sampling_time,i,j]
                        if True in (trans_time[sampling_time:,i,j] > (trans_last_value + self.friendship_num) ):
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

        nx.write_gexf(G,self.path+'%s_graph.gexf'%(graph_type))
#        constructed_graph = G
#        dynamic_graph = tracker.make_dynamic_trans_time_graph(constructed_graph)
#        nx.write_gexf(dynamic_graph,self.path+'dynamic_%s_graph.gexf'%(graph_type))
        self.G = G
        return 
    
    
    def draw_graph_weighted_colored(self,position='spring'):
        plt.figure()
        print("Size of G is:", self.G.number_of_nodes())
#        edgewidth = [ d['weight'] for (u,v,d) in self.G.edges(data=True)]
#        color = [ self.a_matrix[u].situation for u in self.G.nodes()]
        color = list(community.best_partition(self.G).values())
#        size = [self.a_matrix[u].asset*10 for u in self.G.nodes()]
#        size = [ self.a_matrix[u].situation*150 for u in self.G.nodes()]
        size = [ self.a_matrix[u].worth_ratio*150 for u in self.G.nodes()]
        
        if position == 'spring':
            pos = nx.spring_layout(self.G)
        if position == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.G)
        
#        nx.draw(self.G, pos=pos, with_labels = True, node_size=100, font_size=8, width=np.array(edgewidth), node_color=s)
        nx.draw(self.G, pos=pos, with_labels = True, node_size=size, font_size=8, node_color=color, width=0.2)
#        nx.draw(self.G, pos=pos, with_labels = True, node_size=150, font_size=8, width=np.array(edgewidth))
#        plt.savefig(self.path+'graph'+' friedship number:'+str(self.friendship_num))
        plt.savefig(self.path+'graph '+position+' - fpoint %d.png'%(self.friendship_num))
        plt.close()
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
        plt.close()
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
        plt.close()
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
        sigma = (cc/cc_r) / (asph/asph_r)
        omega = asph_r/asph - cc/cc_r
        
        title = 'Topological Charateristics.txt'
        topol_file = open(self.path+title,'w')
        topol_file.write('Size of the Giant Component is: '+str(G0.number_of_nodes())+' with '+str(G0.number_of_edges())+' edges'+'\n')
        topol_file.write('Average Shortert Path Length'+'\n')
        topol_file.write(str(asph)+'\n')
        topol_file.write('Clustering Coeficient'+'\n')
        topol_file.write(str(cc)+'\n')
        topol_file.write('Small-Worldness Sigma'+'\n')
        topol_file.write(str(sigma)+'\n')
        topol_file.write('Small-Worldness Omega'+'\n')
        topol_file.write(str(omega)+'\n')
        topol_file.write('\n')
        topol_file.write('The Corresponding Random Graph Has:'+'\n')
        topol_file.write('Shortert Path Length: '+str(asph_r)+'\n')
        topol_file.write('Clustering Coeficient: '+str(cc_r)+'\n'+'\n')
        topol_file.write('A graph is commonly classified as small-world if Small-Worldness Sigma is bigger than 1. \n\n')
        topol_file.write('The small-world coefficient Omega ranges between -1 and 1.\nValues close to 0 means the G features small-world characteristics.\nValues close to -1 means G has a lattice shape whereas values close to 1 means G is a random graph.')
        topol_file.close()
        return asph, cc, asph_r, cc_r, sigma, omega

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

    def assortativity(self):
        title = 'Assortativity.txt'
        assort_file = open(self.path + title,'w')
        i = 0
        while i not in self.G.nodes():
            i += 1
        for attr in self.G.nodes(data=True)[i].keys():
            assort_file.write('assortativity according to '+attr+' is: '+'\n')
            assort_file.write(str(nx.attribute_assortativity_coefficient(self.G,attr))+'\n'+'\n')
        assort_file.close()
        return 
    
    def money_vs_situation(self,path):
        plt.figure()
        plt.scatter(self.array('situation'),self.array('money'))
        title = 'Money Vs Situation'
        plt.title(title)
        plt.savefig(self.path+title)
        plt.close()
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
            counter = 0
            for i in np.arange(self.N):
                if asset[i] < x[bin_index+1] and x[bin_index] < asset[i]:
                    y[bin_index] += transaction[i]
                    counter += 1
            if counter != 0:
                y[bin_index] /= counter
        plt.figure()
        plt.bar(x[:-1] + width/2,y,width=width)
        title = 'Transaction Vs Asset'
        plt.title(title)
        plt.savefig(self.path+title)
        plt.close()
        return
    
    def rich_club(self,normalized=False):
        if normalized:
            Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
            G0 = self.G.subgraph(Gcc[0])
            rich_club_coef = nx.rich_club_coefficient(G0,normalized=normalized)
            title = 'Rich Club Normalized'
        else:
            rich_club_coef = nx.rich_club_coefficient(self.G,normalized=normalized)
            title = 'Rich Club NonNormalized'
        rc_array = np.zeros(len(rich_club_coef))
        for i in np.arange(len(rich_club_coef)):
            rc_array[i] = rich_club_coef[i]
        plt.figure()
        plt.plot(rc_array)
        plt.title(title)
        plt.savefig(self.path+title)
        plt.close()
        return rc_array

    
    def friendship_point(self,num_transaction,sampling_time):
        """ When we consider someone as friend
        Or in other words: how many transactions one agent with an agent means that they are friends
        """
#        alpha = num_transaction[sampling_time:] / ((1 - explore_prob_arr[sampling_time:]) * self.N)
#        T_eff = np.sum(alpha)
#        beta = 1
#        print('old friendship point',int(np.ceil(beta * T_eff / self.N)))
        
#        avg = np.average(num_transaction[sampling_time:])
        avg = np.average(num_transaction)
        sigma = np.sqrt(np.var(num_transaction))
#        sigma = np.sqrt(np.var(num_transaction[sampling_time:]))
        T_eff = self.T * (avg + 2*sigma)/self.N
        beta = 1
        self.friendship_num = int(np.ceil(beta * T_eff / self.N))
        
        print('friendship point:',self.friendship_num)
        self.transaction_average = avg
        print('average transaction',self.transaction_average)
        return 

    def community_detection(self):
        """Community Detection"""
        community_dict = community.best_partition(self.G)
        partition2 = communityx.greedy_modularity_communities(self.G)
        
        """Modularity"""
        modularity = community.modularity(community_dict,self.G,weight='asdfd')
        coverage = communityx.coverage(self.G,partition2)
        #corresponding random graph
        H = nx.gnm_random_graph(self.G.number_of_nodes(),self.G.number_of_edges())
        part = community.best_partition(H)
        part2 = communityx.greedy_modularity_communities(H)
        modularity_rand = community.modularity(part,H)
        coverage_rand = communityx.coverage(H,part2)
        
        """Write File"""
        title = 'Communities.txt'
        com_file = open(self.path + title,'w')
        com_file.write('Modularity:'+'\n')
        com_file.write(str(modularity)+'\n')
        com_file.write('Coverage'+'\n')
        com_file.write(str(coverage)+'\n')
        com_file.write('The corresponding random graph has modularity:'+'\n')
        com_file.write(str(modularity_rand)+'\n')
        com_file.write('The corresponding random graph has coverage:'+'\n')
        com_file.write(str(coverage_rand))
        com_file.write('\n')
        com_file.write('number of communities:'+'\n')
        com_file.write(str(len(partition2))+'\n')
        com_file.write('\n')
        com_file.write('The coverage of a partition is the ratio of the number of intra-community edges to the total number of edges in the graph.')
        com_file.close()
        return modularity,coverage
    
    def communities_property_hist(self,property_id):
        """properties histogram in inter-communities"""
        community_members = [list(x) for x in communityx.greedy_modularity_communities(self.G)]
        proprety_arr = self.array(property_id)
        communities_property_list = []
        
        for com_members_list in community_members:
            property_list = [ proprety_arr[agent] for agent in com_members_list]
            communities_property_list.append(property_list)
        
        fig, ax = plt.subplots(nrows=1,ncols=1)
        ax.hist(communities_property_list,alpha=0.5)
        ax.set_title('%s in community'%(property_id))
        plt.savefig(self.path + 'inter_com_%s'%(property_id))

#        """Rich Agents in Communities"""
#        for com_num in community_members:
#            community_members_asset = [ self.a_matrix[agent].asset for agent in community_members[com_num] ]
#            community_members[com_num] = [community_members[com_num],community_members_asset]
        
#        richest_in_coms = []
#        for com_num in community_members:
##            richest_in_coms = community_members[com_num][0][ np.argsort(community_members[com_num][1])[0] ]
#            richest_index = np.where(community_members[com_num][1] == max(community_members[com_num][1]))[0][0]
#            richest_in_coms.append(community_members[com_num][0][richest_index])
#        return  richest_in_coms
        return


    def graph_correlations(self):
        nodes = self.G.nodes()
        nodes_dict = dict(self.G.nodes(data=True))
        i = 0
        while i not in nodes:
            i += 1
        attributes = nodes_dict[i].keys()
        length = len(attributes)
        correlation = np.zeros((length,length))
        attr_array = np.zeros((length,len(nodes_dict)))
        attr_array_avg = np.zeros(length)
        attr_index = 0
        for attr in attributes:
            i = 0
            for n in nodes:
                attr_array[attr_index,i] = nodes_dict[n][attr]
                i += 1
            attr_index += 1
        attr_array_avg = np.average(attr_array,axis=1)
        for i in np.arange(length):
            for j in np.arange(length):
                if j > i:
                    numerator = np.sum( (attr_array[i,:]-attr_array_avg[i])*(attr_array[j,:]-attr_array_avg[j]))
                    denominator = np.sqrt(np.sum( (attr_array[i,:]-attr_array_avg[i])**2 ) * np.sum( (attr_array[j,:]-attr_array_avg[j])**2 ) )
                    correlation[i,j] = numerator / denominator
                elif j < i:
                    correlation[i,j] = correlation[j,i]
                elif j==i:
                    correlation[i,j] = 1
        fig, ax = plt.subplots(nrows=1,ncols=1)
        im = ax.imshow(correlation)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('correlation', rotation=-90, va="bottom")
        title = ''
        for attr in attributes:
            title += attr + ', '
        plt.title(title)
        plt.savefig(self.path + 'correlation in graph')
        plt.close()
        return

    
    def graph_related_chars(self,num_transaction,tracker):
        path = self.path
        try:
            os.mkdir(path + '\\graph_related')
        except:
            print('exists')
        dic = {'modul':[],'cover':[],'asph':[],'asph_r':[],'cc':[],'cc_r':[],'sigma':[],'omega':[],'rc':[]}
        for i in np.arange(20):
            try:
                for arr in dic:
                    dic[arr].append(0)
                self.path = path + '\\graph_related' + '\\{0}, '.format(i)
                self.graph_construction('trans_number',num_transaction,boolean=False,fpoint=i,tracker_obj=tracker)
                self.draw_graph_weighted_colored('spring')
                self.draw_graph_weighted_colored('kamada_kawai')
                self.hist('degree')
                self.hist_log_log('degree')
                dic['modul'][i], dic['cover'][i] = self.community_detection()
                dic['asph'][i], dic['cc'][i], dic['asph_r'][i], dic['cc_r'][i], dic['sigma'][i], dic['omega'][i] = self.topology_chars()
                self.assortativity()
                self.graph_correlations()
                dic['rc'][i] = self.rich_club()
            except:
                print('cannot make more graphs',i)
                break
        
        """Plot"""
        self.plot_general(path,dic['modul'],title='GR Modularity Vs Friendship Point')
        self.plot_general(path,dic['cover'],title='GR Coverage Vs Friendship Point')
        self.plot_general(path,dic['sigma'],title='GR Smallworldness Sigma Vs Friendship Point')
        self.plot_general(path,dic['omega'],title='GR Smallworldness Omega Vs Friendship Point')
        self.plot_general(path,dic['cc'],second_array=dic['cc_r'],title='GR Clustering Coefficient Vs Friendship Point')
        self.plot_general(path,dic['asph'],second_array=dic['asph_r'],title='GR Shortest Path Length Vs Friendship Point')
        self.plot_general(path,np.array(dic['cc'])/np.array(dic['cc_r']),title='GR Clustering Coefficient Normalized Vs Friendship Point')
        self.plot_general(path,np.array(dic['asph'])/np.array(dic['asph_r']),title='GR Shortest Path Length Normalized Vs Friendship Point')
        self.plot_general(path,dic['rc'],indicator=False,title='GR Rich Club Vs Friendship Point')
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

    
class Tracker: #XXX
    
    def __init__(self,number_agent,total_time,size,a_matrix):
        
        self.a_matrix = a_matrix
        self.T = total_time
        self.memory_size = size
        self.N = number_agent
        if self.T >= 1000:
            sampling_time = 1000
        else:
            sampling_time = int(self.T / 2)

        """Trackers"""
        self.self_value = np.zeros((self.T,self.N))
        self.valuable_to_others = np.zeros((self.T,self.N))
        self.worth_ratio = np.zeros((self.T-2,self.N))
        self.trans_time = np.ones((sampling_time,self.N,self.N))
        self.correlation_mon = np.zeros(self.T)
        self.correlation_situ = np.zeros(self.T)
        
        self.agents_money  = np.zeros((self.T,self.N))
        self.agents_asset  = np.zeros((self.T,self.N))
        self.agents_approval  = np.zeros((self.T,self.N))
        
    def update_A(self,a_matrix):
        self.a_matrix = a_matrix
        return
    
    def get_path(self,path):
        self.path = path
        return
        
    def get_list(self,get_list,t):
        
        if get_list == 'self_value':
            self.self_value[t] = np.sum(self._array('value'),axis = 1)
        if get_list == 'valuable_to_others':
            self.valuable_to_others[t] = np.sum(self._array('value'),axis = 0)
        if get_list == 'worth_ratio':
            self.worth_ratio[t] = self._array('worth_ratio')
            
        if get_list == 'money':
            self.agents_money[t] = self._array('money')
        if get_list == 'asset':
            self.agents_asset[t] = self._array('asset')
        if get_list == 'asset':
            self.agents_approval[t] = self._array('approval')
        
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
        
        if what_array == 'situation':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].situation
            return ref[what_array]
        
        if what_array == 'asset':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].asset
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
    
    def plot_general(self,array,title=''):
        plt.figure()
        plt.plot(array)
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
        plt.title(title+' histogram log-log'+' N={} T={}'.format(self.N,self.T))
        plt.savefig(self.path+title+' histogram log-log'+' N={} T={}'.format(self.N,self.T))
        plt.close()
        return
    
    def index_in_arr(array,value):
        return np.where( array == value )[0][0]
    
    def trans_time_visualizer(self,agent_to_watch,title,**kwargs):
        """
        it will show each node transaction transcript.
        """
        sort_by = kwargs.get('sorting_feature','situation')
        
        fig, ax = plt.subplots(nrows=1,ncols=1)
        
        sort_arr = self._array(sort_by)
#        sort_arr_sorted = np.sort(sort_arr)
#        x_label_list = ['%.2f'%(sort_arr_sorted[i]) for i in range(self.N) ]
#        ax.set_xticklabels(x_label_list)
        
        im = ax.imshow(self.trans_time[:,agent_to_watch,np.argsort(sort_arr)].astype(float),aspect='auto')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('number of transactions', rotation=-90, va="bottom")
        plt.title(title+' asset:{0:.3g}'.format(self.a_matrix[agent_to_watch].asset)+'of agent {0} with {1:.2f} situation'.format(agent_to_watch,self.a_matrix[agent_to_watch].situation))
        plt.savefig(self.path+title)
        plt.close()
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
    
    def correlation_growth_situation(self,survey_property_id,base_property_id):
        
        survey_ref = {'money':self.agents_money,'approval':self.agents_approval,'asset':self.agents_asset}
        base_ref = { base_property_id:self._array(base_property_id)}
        
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
        
    def valuability(self):
        fig, ax = plt.subplots(nrows=1,ncols=1)
        asset = self._array('asset')
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
    
    def property_evolution(self,property_id):
        ref = {'money':self.agents_money,'approval':self.agents_approval,'asset':self.agents_asset}
        property_arr = ref[property_id]
        
        fig1, (ax1,ax2) = plt.subplots(nrows=2,ncols=1)
        fig2, (ax3,ax4) = plt.subplots(nrows=2,ncols=1)
        
        ax1.title.set_text('last&first ' + property_id + ' vs situation')
        ax1.scatter(self._array('situation'),property_arr[0,:],c='r')
        
        for t in np.arange(1,self.T,self.T-1,dtype = int):
            ax1.scatter(self._array('situation'),property_arr[t,:])
        
        ax2.title.set_text(property_id + ' growth vs situation')
        ax2.scatter(self._array('situation'),property_arr[self.T-1,:] - property_arr[0,:])
        
        ax3.title.set_text('initial vs last'+property_id)
        ax3.scatter(property_arr[0,:],property_arr[self.T-1,:])
        ax4.title.set_text(property_id+' growth')
        ax4.scatter(property_arr[0,:],property_arr[self.T-1,:] - property_arr[0,:])
        
        fig1.savefig(self.path + property_id + ' growth vs situation')
        fig2.savefig(self.path + 'initial vs last'+property_id)
        return
    

    
    