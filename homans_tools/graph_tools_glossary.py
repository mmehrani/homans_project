"""
Created on Tue Dec 17 11:41:02 2019

@author: Mohsen Mehrani, Taha Enayat
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import community
from networkx.algorithms import community as communityx
from agents_properties_tools import arrays_glossary
import matplotlib.cm as cm
import sys
pd = {'win32':'\\', 'linux':'/'}
if sys.platform.startswith('win32'):
    plat = 'win32'
elif sys.platform.startswith('linux'):
    plat = 'linux'

class Community_related_tools():
    
#    def __init__(self):
#        self.modularity_communities = None
#        self.best_parts = None
    
    def community_division(self):
        """
        it just returns the community division subsets.
        """
        return communityx.greedy_modularity_communities(self.G)
    
    def assign_communities(self):
        """
        assigned communities to be calculated just once for a graph in all analysis funcs.
        """
#        self.modularity_communities = communityx.greedy_modularity_communities(self.G)
#        community_members = [list(x) for x in self.modularity_communities]
#        com_dict = {}
#        for i,com in enumerate(community_members):
#            for node in com:
#                com_dict[node] = i
#
#        self.best_parts = com_dict
        
        self.modularity_communities = communityx.greedy_modularity_communities(self.G)
        self.best_parts = community.best_partition(self.G)
        return
    
    def community_detection(self):
        """Community Detection"""
        community_dict = self.best_parts
#        partition2 = self.partition
        partition2 = self.modularity_communities
        
        """Modularity"""
        modularity = community.modularity(community_dict,self.G)
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
#        com_file.write(str(len(partition2))+'\n')
        com_file.write(str(max(self.best_parts.values())+1)+'\n')
        com_file.write('\n')
        com_file.write('The coverage of a partition is the ratio of the number of intra-community edges to the total number of edges in the graph.')
        com_file.close()
        return modularity,coverage,modularity_rand,coverage_rand
    
    def communities_property_hist(self,property_id,boolean=True):
        """properties histogram in inter-communities"""
        community_members = [list(x) for x in self.modularity_communities]
        proprety_arr = self.array(property_id)
        communities_property_list = []
        
        for com_members_list in community_members:
            property_list = [ proprety_arr[agent] for agent in com_members_list]
            communities_property_list.append(property_list)
        
        if boolean:
            fig, ax = plt.subplots(nrows=1,ncols=1)
            ax.hist(communities_property_list,alpha=0.5)
            ax.set_title('%s in community'%(property_id))
            plt.savefig(self.path + 'C inter community %s'%(property_id))
            plt.close()
        
        property_sum=[];property_mean=[];property_var=[];
        for com_prop_list in communities_property_list:
            summ = sum(com_prop_list)
            mean = sum(com_prop_list) / len(com_prop_list)
            var = np.sqrt( sum( (com_prop_list - mean)**2 ) / len(com_prop_list) )
            property_sum.append(summ)
            property_mean.append(mean)
            property_var.append(var)
        
        cmap = cm.ScalarMappable()
        plt.figure()
        plt.bar( property_mean, property_sum, width=property_var, alpha=0.5, color= cmap.to_rgba(np.arange(len(property_sum))) )
        plt.title('{} in community'.format(property_id))
        plt.savefig(self.path + 'C community overal {}'.format(property_id))
        plt.close()
        return
    
    def communities_property_evolution(self,tracker,property_id):
        """communities asset growth"""
        survey_ref = {'money':tracker.agents_money,
                      'approval':tracker.agents_approval,
                      'asset':tracker.agents_asset}
        survey_arr = survey_ref[property_id]
        
        community_dict = self.community_division()
        community_dict = [list(x) for x in community_dict]
        
        communities_property_evolution_list = []
        communities_property_evolution_list_err = []
        for subset in community_dict:
            total_evo_property = np.sum( survey_arr[:,subset] ,axis = 1)
            var_evo_property = np.var( survey_arr[:,subset] ,axis = 1)
            
            communities_property_evolution_list.append(total_evo_property)
            communities_property_evolution_list_err.append(var_evo_property)
        
        fig, ax = plt.subplots(nrows=1,ncols=1)
        for i in range(len(communities_property_evolution_list)):
            ax.errorbar(np.arange(self.T),communities_property_evolution_list[i],yerr = communities_property_evolution_list_err[i])
        ax.set_title('%s evolution of communities'%(property_id))
        plt.savefig(self.path + 'C community %s evolution'%(property_id))
        return
    pass

class Graph_related_tools(arrays_glossary,Community_related_tools):
    def __init__(self,current_time,number_agent,a_matrix,**kwargs):
        self.a_matrix = a_matrix
        self.N = number_agent
        self.T = current_time #should be overrided in analysis class
        self.path = kwargs.get('alter_path',None)
        return
    def graph_construction(self,graph_type,num_transaction,sample_time_trans,boolean=True,**kwargs):
#        time = kwargs.get('time',self.T)
        G = nx.Graph()
        if graph_type == 'trans_number':
            sampling_time = kwargs.get('sampling_time',2000)
        
#            sampling_time = 80
#            if sampling_time > self.T:
#                sampling_time = self.T
                
#            trans_time = kwargs.get('trans_time',None)
#            sample_time_trans = kwargs.get('sample_time_trans',None)
#            tracker = Tracker(self.N,self.T,self.memory_size,self.a_matrix)
            if boolean:
                self.friendship_point(num_transaction,sampling_time)
#                self.friendship_point(num_transaction)
            else:
                self.friendship_num = kwargs.get('fpoint')
#            if trans_time != None or sample_time_trans != None:
#            if type(sample_time_trans) != None:
            for i in np.arange(self.N):
                for j in self.a_matrix[i].active_neighbor.keys():
#                        trans_last_value = tracker.trans_time[sampling_time,i,j]
#                        trans_last_value = trans_time[sampling_time,i,j]
                    trans_last_value = sample_time_trans[i,j]
#                        if True in (trans_time[sampling_time:,i,j] > (trans_last_value + self.friendship_num) ):
                    if (self.a_matrix[i].neighbor[j] >= (trans_last_value + self.friendship_num) ):
                        G.add_edge(i,j)
                        
        node_attr_dict = { i:{'situation':0,'money':0,'worth_ratio':0,'others_feeling':0} for i in G.nodes() }
        for i in G.nodes():
            node_attr_dict[i]['situation'] = float(self.a_matrix[i].situation)
            node_attr_dict[i]['money'] = float(self.a_matrix[i].money)
            node_attr_dict[i]['worth_ratio'] = float(self.a_matrix[i].worth_ratio)
            node_attr_dict[i]['approval'] = float(self.a_matrix[i].approval)
            node_attr_dict[i]['others_feeling'] = float(self.array('others_feeling')[i])
            node_attr_dict[i]['asset'] = float(self.a_matrix[i].asset)
        
        nx.set_node_attributes(G,node_attr_dict)
        
        if self.path != None:
            nx.write_gexf(G,self.path+'%s_%d_graph.gexf'%(graph_type,self.T))
#        constructed_graph = G
#        dynamic_graph = tracker.make_dynamic_trans_time_graph(constructed_graph)
#        nx.write_gexf(dynamic_graph,self.path+'dynamic_%s_graph.gexf'%(graph_type))
        self.G = G
        self.assign_communities()
        return G
    
    
    def draw_graph_weighted_colored(self,position='spring'):
        plt.figure()
        print("Size of G is:", self.G.number_of_nodes())
#        edgewidth = [ d['weight'] for (u,v,d) in self.G.edges(data=True)]
#        color = [ self.a_matrix[u].situation for u in self.G.nodes()]
        color = list(self.best_parts.values())
        size = [self.a_matrix[u].asset*15 for u in self.G.nodes()]
#        size = [self.a_matrix[u].money*30 for u in self.G.nodes()]
#        size = [ self.a_matrix[u].situation*150 for u in self.G.nodes()]
#        size = [ self.a_matrix[u].worth_ratio*150 for u in self.G.nodes()]
        
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

    
    def friendship_point(self,num_transaction,sampling_time):
        """ When we consider someone as friend
        Or in other words: how many transactions one agent with an agent means that they are friends
        """
#        alpha = num_transaction[sampling_time:] / ((1 - explore_prob_arr[sampling_time:]) * self.N)
#        T_eff = np.sum(alpha)
#        beta = 1
#        print('old friendship point',int(np.ceil(beta * T_eff / self.N)))
        
#        avg = np.average(num_transaction[sampling_time:])
        avg = np.average(num_transaction) #num trans has been saved due to sampling time
        sigma = np.sqrt(np.var(num_transaction))
#        sigma = np.sqrt(np.var(num_transaction[sampling_time:]))
#        T_eff = self.T * (avg + 2*sigma)/self.N
        T_eff = sampling_time * (avg + 2*sigma)/self.N
        beta = 1

        self.friendship_num = int(np.ceil(beta * T_eff / self.N))
        
        print('friendship point:',self.friendship_num)
        self.transaction_average = avg
        print('average transaction',self.transaction_average)
        return 
    
    def graph_correlations(self,all_nodes = False):
        nodes = self.G.nodes()
        nodes_dict = dict(self.G.nodes(data=True))
        i = 0
        while i not in nodes:
            i += 1
        attributes = nodes_dict[i].keys()
        length = len(attributes)
        
        if all_nodes == True:
            nodes_not_in_graph = []
            for node in range(self.N):
                if node not in nodes:
                    nodes = np.append(nodes,[[node]])
                    nodes_dict[node] = {attr:self.array(attr)[node] for attr in attributes}
                    
        
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
#        attr_array_avg = np.average(attr_array,axis=1)
#        for i in np.arange(length):
#            for j in np.arange(length):
#                if j > i:
#                    numerator = np.sum( (attr_array[i,:]-attr_array_avg[i])*(attr_array[j,:]-attr_array_avg[j]))
#                    denominator = np.sqrt(np.sum( (attr_array[i,:]-attr_array_avg[i])**2 ) * np.sum( (attr_array[j,:]-attr_array_avg[j])**2 ) )
#                    correlation[i,j] = numerator / denominator
#                elif j < i:
#                    correlation[i,j] = correlation[j,i]
#                elif j==i:
#                    correlation[i,j] = 1
        correlation = np.corrcoef(attr_array)                
        fig, ax = plt.subplots(nrows=1,ncols=1)
        im = ax.imshow(correlation)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('correlation', rotation=-90, va="bottom")
#        title = ''
#        for attr in attributes:
#            title += attr + ', '
#        plt.title(title)
        
        attributes = list(attributes)
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(attributes)))
        ax.set_yticks(np.arange(len(attributes)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(attributes)
        ax.set_yticklabels(attributes)
        
        ax.set_ylim(-0.5,len(attributes)-0.5)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(attributes)):
            for j in range(len(attributes)):
                text = ax.text(j, i, "%.2f"%(correlation[i, j]),
                               ha="center", va="center", color="w")
        fig.tight_layout()
        
        if all_nodes == True:
            status = 'all nodes'
        else:
            status = 'graph nodes'
            
        plt.savefig(self.path + 'correlation in network ' + status)
        plt.close()
        return
    
    def graph_related_chars(self,num_transaction,tracker,sampling_time):
        path = self.path
        try:
            os.mkdir(path +'graph_related')
        except:
            print('exists')
        dic = {'modul':[],'cover':[],'asph':[],'asph_r':[],'cc':[],'cc_r':[],'sigma':[],'omega':[],'rc':[],'cover_r':[],'modul_r':[],'nsize':[],'esize':[]}

        local0,local1,local2,local3 = -1,-1,-1,-1
        for i in np.arange(20)+1:
            print('i =',i)
            self.path = path + 'graph_related' + pd[plat]+'{0}, '.format(i)
            self.graph_construction('trans_number',num_transaction,boolean=False,fpoint=i,sampling_time=sampling_time,sample_time_trans=tracker.sample_time_trans)
            if self.G.number_of_nodes() == 0:
                print('cannot make more graphs')
                self.graph_construction('trans_number',num_transaction,sampling_time=sampling_time,sample_time_trans=tracker.sample_time_trans)
                break
            self.draw_graph_weighted_colored('spring')
            self.draw_graph_weighted_colored('kamada_kawai')
            try:
                local0 += 1
                dic['nsize'].append(0)
                dic['esize'].append(0)
                dic['nsize'][local0] = self.G.number_of_nodes()
                dic['esize'][local0] = self.G.number_of_edges()
            except: print('graph size')
            try:
                self.hist('degree')
                self.hist_log_log('degree')
            except: print('degree hist')
            try:
                local1 += 1
                dic['modul'].append(0)
                dic['cover'].append(0)
                dic['modul_r'].append(0)
                dic['cover_r'].append(0)
                dic['modul'][local1], dic['cover'][local1],dic['modul_r'][local1], dic['cover_r'][local1] = self.community_detection()
            except: print('community detection')
            try:
                local2 += 1
                dic['asph'].append(0)
                dic['cc'].append(0)
                dic['asph_r'].append(0)
                dic['cc_r'].append(0)
                dic['sigma'].append(0)
                dic['omega'].append(0)
                dic['asph'][local2], dic['cc'][local2], dic['asph_r'][local2], dic['cc_r'][local2], dic['sigma'][local2], dic['omega'][local2] = self.topology_chars()
            except: print('topology chars')
            try:
                self.assortativity()
            except: print('assortativity')
            try:
                self.graph_correlations()
            except: print('correlations')
            try:
                local3 += 1
                dic['rc'].append(0)
                dic['rc'][local3] = self.rich_club()
            except: print('rich club')
            try:
                for prop in ['money','asset','approval','worth_ratio','situation']:
                    self.communities_property_hist(prop,boolean=False)
            except: print('property hist')

        self.path = path
        """Plot"""
        self.plot_general(path,dic['modul'],second_array=dic['modul_r'],title='GR Modularity Vs Friendship Point')
        self.plot_general(path,dic['cover'],second_array=dic['cover_r'],title='GR Coverage Vs Friendship Point')
        self.plot_general(path,dic['sigma'],title='GR Smallworldness Sigma Vs Friendship Point')
        self.plot_general(path,dic['omega'],title='GR Smallworldness Omega Vs Friendship Point')
        self.plot_general(path,dic['cc'],second_array=dic['cc_r'],title='GR Clustering Coefficient Vs Friendship Point')
        self.plot_general(path,dic['asph'],second_array=dic['asph_r'],title='GR Shortest Path Length Vs Friendship Point')
        self.plot_general(path,np.array(dic['cc'])/np.array(dic['cc_r']),title='GR Clustering Coefficient Normalized Vs Friendship Point')
        self.plot_general(path,np.array(dic['asph'])/np.array(dic['asph_r']),title='GR Shortest Path Length Normalized Vs Friendship Point')
        self.plot_general(path,dic['rc'],indicator=False,title='GR Rich Club Vs Friendship Point')
        self.plot_general(path,dic['nsize'],title='GR Number of Nodes in Each Friendship Point')
        self.plot_general(path,dic['esize'],title='GR Number of Edges in Each Friendship Point')
        return
    
    pass



