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
    
    def assign_communities(self):
        """
        assigned communities to be calculated just once for a graph in all analysis funcs.
        
        It generates communities from both networkx community and python louvain community.
        In the code we mostly use the latter as community so it is necessary to first 
        pip the python-louvain and community.
        
        However, user may swich to networkx community by commenting and decommenting 
        few lines. These lines are marked by #XXX, although coloring the graph is only
        available with python-louvain community.
        """
        self.modularity_communitiesx = [list(x) for x in communityx.greedy_modularity_communities(self.G)]
        self.best_parts = community.best_partition(self.G)
        com_dict = {}
        for i,com in enumerate(self.modularity_communitiesx):
            for node in com:
                com_dict[node] = i
        
        com_list=[[] for c in list(set(self.best_parts.values()))]
        for n,c in zip(self.best_parts.keys(),self.best_parts.values()):
            com_list[c].append(n)
        self.modularity_communities = com_list
        self.best_parts_x = com_dict
        return
    
    def community_detection(self):
        """
        Detects characteristics related to communities of graph and writes them 
        down to the 'Communities.txt' file. It also compares these characteristics
        with a random graph of the same node-size and edge-size.
        """
        partitionx = communityx.greedy_modularity_communities(self.G)
        
        """Modularity & Coverage"""
        modularity = community.modularity(self.best_parts,self.G) #XXX
        coverage = communityx.coverage(self.G,partitionx)
        
        """in the corresponding random graph"""
        H = nx.gnm_random_graph(self.G.number_of_nodes(),self.G.number_of_edges())
        part = community.best_partition(H) #XXX
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
        com_file.write(str(max(self.best_parts.values())+1)+'\n') #XXX
        # com_file.write(str(max(self.best_parts_x.values())+1)+'\n')
        com_file.write('\n')
        com_file.write('The coverage of a partition is the ratio of the number of intra-community edges to the total number of edges in the graph.')
        com_file.close()
        return modularity, coverage, modularity_rand, coverage_rand
    
    def communities_property_dist(self,property_id,boolean=True):
        """
        Creates a unique figure indicating the distribution of properties in communities. 
        property_id is the property which this function generates its distribution.
        
        Color of each bar is community; that is each community has different color.
        Height of each bar is the number of members within that community.
        Width of each bar is the standard deviation of the property.
        Middle of each bar is located at the average of the property.
        
        For example:
            we want to compute asset distribution of communities.
            community 1 has 20 members and average of their assets is 6 and the
            standard deviation of this community is 2. 
            So in the figure we see a bar from [4,8] and the height of 20.
        """
        community_members = self.modularity_communities #XXX
        # community_members = self.modularity_communitiesx
        
        proprety_arr = self.array(property_id)
        communities_property_list = []
        community_no = []
        
        for com_members_list in community_members:
            property_list = [ proprety_arr[agent] for agent in com_members_list]
            communities_property_list.append(property_list)
            community_no.append(len(com_members_list))
        
        if boolean:
            fig, ax = plt.subplots(nrows=1,ncols=1)
            ax.hist(communities_property_list,alpha=0.5)
            ax.set_title('%s in community'%(property_id))
            plt.savefig(self.path + 'C inter community %s'%(property_id))
            plt.close()
        
        property_sum,property_mean,property_var = [],[],[]
        for com_prop_list in communities_property_list:
            summ = sum(com_prop_list)
            mean = sum(com_prop_list) / len(com_prop_list)
            var = np.sqrt( sum( (com_prop_list - mean)**2 ) / len(com_prop_list) )
            property_sum.append(summ)
            property_mean.append(mean)
            property_var.append(var)
            
        cmap = cm.ScalarMappable()
        plt.figure()
#        plt.bar( property_mean, property_sum, width=property_var, alpha=0.5, color= cmap.to_rgba(np.arange(len(property_sum))) )
        plt.bar( property_mean, community_no, width=property_var, alpha=0.5, color= cmap.to_rgba(np.arange(len(property_sum))) )
        plt.title('{} in community'.format(property_id))
        plt.savefig(self.path + 'C community overal {}'.format(property_id))
        plt.close()
        return
    
    def communities_property_evolution(self,tracker,property_id):
        """
        Evolution of selected property in different communities throughout the time.
        """
        survey_ref = {'money':tracker.agents_money,
                      'approval':tracker.agents_approval,
                      'asset':tracker.agents_asset}
        survey_arr = survey_ref[property_id]
        
        community_dict = self.modularity_communities #XXX
        # community_dict = self.modularity_communitiesx
        
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
        """
        Makes the graph by this definition:
            if two given agents have transacted within the interval of [T - sampling_time, T],
            more than certain times, they are cosidered as friends and a link establishes 
            between them. friendship_point function decides how many transaction is needed to be
            considered as friend.

        Parameters
        ----------
        graph_type : string
            Indicates the definition of graph which is 'trans_number'.
        num_transaction : 1D-array of size T
            used as an argument to friendship_point function.
        sample_time_trans : 2D-array of size N*N
            data of transaction are in this variable.
        boolean : TYPE, optional
            When graph_related function is called, it changes to False so that we 
            have control over friendship_num. The default is True.
        **kwargs : TYPE
            sampling_time which is a number & fpoint (when this function is 
            called in graph_related func.

        Returns
        -------
        Makes graph and calls assign_communities function.
        """
        
        """ Graph Making """
        G = nx.Graph()
        if graph_type == 'trans_number':
            sampling_time = kwargs.get('sampling_time',1000)

            if boolean:
                self.friendship_point(num_transaction,sampling_time)
            else:
                self.friendship_num = kwargs.get('fpoint')

            for i in np.arange(self.N):
                for j in self.a_matrix[i].active_neighbor.keys():
                    trans_last_value = sample_time_trans[i,j]
                    if (self.a_matrix[i].neighbor[j] >= (trans_last_value + self.friendship_num) ):
                        G.add_edge(i,j)
                    
        """ Adding attributes """
        node_attr_dict = { i:{'asset':0,'money':0,'approval':0,'situation':0,'worth_ratio':0} for i in G.nodes() }
        for i in G.nodes():
            node_attr_dict[i]['situation'] = int(self.a_matrix[i].situation * 20)
            node_attr_dict[i]['money'] = int(self.a_matrix[i].money * 100)
            node_attr_dict[i]['approval'] = int(self.a_matrix[i].approval * 100)
            node_attr_dict[i]['asset'] = int(self.a_matrix[i].asset * 100)
            node_attr_dict[i]['worth_ratio'] = int(self.a_matrix[i].worth_ratio * 100)
        
        nx.set_node_attributes(G,node_attr_dict)
        
        """ Creating Ghephi file """
        if self.path != None:
            nx.write_gexf(G,self.path+'Gephi graph T={0} sampling_time={1}.gexf'.format(self.T,sampling_time))

        self.G = G
        self.assign_communities()
        return G
    
    def draw_graph_weighted_colored(self,position='spring',nsize='asset',ncolor='community'):
        """ 
        Draws graph with giver position, node size, and node color character 
        """
        plt.figure()
        print("Size of G is:", self.G.number_of_nodes())

        if nsize == 'asset':
            size = [self.a_matrix[u].asset*15 for u in self.G.nodes()]
        if nsize == 'money':
            size = [self.a_matrix[u].money*30 for u in self.G.nodes()]
        if nsize == 'approval':
            size = [self.a_matrix[u].approval*30 for u in self.G.nodes()]
        if nsize == 'degree':
            size = np.array(list(self.G.degree))*5
        
        if ncolor =='community':
            color = list(self.best_parts.values())
        if ncolor =='situation':
            color = [ self.a_matrix[u].situation for u in self.G.nodes()]
        if ncolor =='worth_ratio':
            color = [ self.a_matrix[u].worth_ratio for u in self.G.nodes()]
        
        if position == 'spring':
            pos = nx.spring_layout(self.G)
        if position == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.G)
        
        nx.draw(self.G, pos=pos, with_labels = True, node_size=size, font_size=8, node_color=color, width=0.1)
        plt.savefig(self.path+'graph '+position+' fpoint=%d'%(self.friendship_num)+' s='+nsize+' c='+ncolor)
        plt.close()
        return
    
    def topology_chars(self):
        """
        Calculates clustering coefficient, average shortest path length, small-worldness and
        compares them with random graph.
        """

        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        G0 = self.G.subgraph(Gcc[0])
        H = nx.gnm_random_graph(self.G.number_of_nodes(),self.G.number_of_edges()) #for clustering
        F = nx.configuration_model([d for v, d in self.G.degree()]) #for shortest path length
        Fcc = sorted(nx.connected_components(F), key=len, reverse=True)
        F0 = F.subgraph(Fcc[0])
        cc = nx.average_clustering(self.G)
        cc_r = nx.average_clustering(H)
        asph = nx.average_shortest_path_length(G0)
        asph_r = nx.average_shortest_path_length(F0)
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
    
    def rich_club(self,normalized=True):
        """ 
        Computes Rich club coefficient of network.
        """
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        G0 = self.G.subgraph(Gcc[0])
        if normalized:
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
    
    def assortativity(self,boolean=True):
        """ 
        Computes assortativity according to network's attributes. 
        It also computes assortativity of these attributes in each community.
        """
        title = 'Assortativity.txt'
        assort_file = open(self.path + title,'w')
        i = 0
        while i not in self.G.nodes():
            i += 1
        attributes = self.G.nodes(data=True)[i].keys()
        assorted = np.zeros(len(attributes)) #it can be made bigger so that it includes communities
        
        for i,attr in enumerate(attributes):
            assort_file.write('assortativity according to '+attr+' is: '+'\n')
            assorted_attr = nx.numeric_assortativity_coefficient(self.G,attr)
            assorted[i] = assorted_attr
            assort_file.write(str(assorted_attr)+'\n\n')
        
        if boolean:     #it does not computes in graph_related
            community_members = self.modularity_communities #XXX
            # community_members = self.modularity_communitiesx
            
            for num,com in enumerate(community_members):
                assort_file.write('community #{}:'.format(num)+'\n')
                for attr in self.G.nodes(data=True)[i].keys():
                    assort_file.write('assortativity according to '+attr+' is: '+'\n')
                    assort_file.write(str(nx.numeric_assortativity_coefficient(self.G,attr,nodes = com))+'\n'+'\n')
            
        assort_file.close()
        return assorted, attributes
    
    def friendship_point(self,num_transaction,sampling_time):
        """ When we consider someone as friend
        Or in other words: how many transactions one agent with an agent means that they are friends
        """
        avg = np.average(num_transaction) #num trans has been saved due to sampling time
        sigma = np.sqrt(np.var(num_transaction))
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
        attributes = list(nodes_dict[i].keys())
        length = len(attributes)
        
        if all_nodes == True:
            for node in range(self.N):
                if node not in nodes:
                    nodes = np.append(nodes,[[node]])
                    nodes_dict[node] = {attr:self.array(attr)[node] for attr in attributes}
                    
        correlation = np.zeros((length,length))
        attr_array = np.zeros((length,len(nodes_dict)))
        attr_index = 0
        for attr in attributes:
            i = 0
            for n in nodes:
                attr_array[attr_index,i] = nodes_dict[n][attr]
                i += 1
            attr_index += 1

        correlation = np.corrcoef(attr_array)                
        fig, ax = plt.subplots(nrows=1,ncols=1)
        im = ax.imshow(correlation)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('correlation', rotation=-90, va="bottom")
        
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
                ax.text(j, i, "%.2f"%(correlation[i, j]),
                               ha="center", va="center", color="w")
        fig.tight_layout()
        if all_nodes == True:
            status = 'all nodes'
        else:
            status = 'graph nodes'
            
        plt.savefig(self.path + 'correlation in network ' + status)
        plt.close()
        return
    
    def intercommunity_links(self):
        """ 
        Generates a figure which each square indicates the ratio of actual existing
        edges to potential edges between communities.
        
        For example if the number of square[1,3] is 0.1, it means that in the network
        between community 1 and 3, there exists 1/10 of edges that can potentially exist.
        """
        community_members = self.modularity_communities #XXX
        # community_members = self.modularity_communitiesx
        
        length = len(community_members)
        edge_arr = np.zeros((length,length))
        for com_num1,comm1 in enumerate(community_members):
            for com_num2,comm2 in enumerate(community_members):
                for mem1 in comm1:
                    for mem2 in comm2:
                        if self.G.has_edge(mem1,mem2):
                            edge_arr[com_num1,com_num2] += 1
        for com_num1,comm1 in enumerate(community_members):
            len1 = len(comm1)
            for com_num2,comm2 in enumerate(community_members):
                len2 = len(comm2)
                if com_num1 == com_num2:
                    edge_arr[com_num1,com_num2] /= (len1 * (len1-1))
                if com_num1 < com_num2:
                    edge_arr[com_num1,com_num2] /= (len1 * len2)
                if com_num1 > com_num2:
                    edge_arr[com_num1,com_num2] = edge_arr[com_num2,com_num1]
        fig, ax = plt.subplots(nrows=1,ncols=1)
        im = ax.imshow(edge_arr)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Edge Ratio', rotation=-90, va="bottom")
        
        for i in np.arange(length):
            for j in np.arange(length):
                ax.text(j, i, "{:.2f}".format(edge_arr[i, j]),ha="center", va="center", color="w")
        
        title = 'Edge Distribution Inter and Intra Community'
        plt.title(title)
        plt.savefig(self.path + title)
        plt.close()
        return 
    
    def graph_related_chars(self,num_transaction,tracker,sampling_time):
        """ 
        Function which combines data from networks with different friendship number.
        
        It stores all data for each friendship number and at the end combines these data.
        """
        path = self.path
        try:
            os.mkdir(path +'graph_related')
        except:
            print('exists')
        dic = {'modul':[],'cover':[],'asph':[],'asph_r':[],'cc':[],
               'cc_r':[],'sigma':[],'omega':[],'rc':[],'rc_la':[],'cover_r':[],
               'modul_r':[],'nsize':[],'esize':[],'is_con':[],'assort':[]}

        local0,local1,local2,local3,local4 = -1,-1,-1,-1,-1
        length = 30
        for i in np.arange(length)+1:
            print('i =',i)
            self.path = path + 'graph_related' + pd[plat]+'{0}, '.format(i)
            try:
                self.graph_construction('trans_number',num_transaction,boolean=False,fpoint=i,sampling_time=sampling_time,sample_time_trans=tracker.sample_time_trans)
                self.draw_graph_weighted_colored(position='spring',)
                self.draw_graph_weighted_colored(position='spring',nsize='money',ncolor='situation')
                self.draw_graph_weighted_colored(position='kamada_kawai')
                self.draw_graph_weighted_colored(position='kamada_kawai',nsize='money',ncolor='situation')
            except: break
            try:
                temp0, temp1, temp2 = self.G.number_of_nodes(), self.G.number_of_edges(), nx.is_connected(self.G)
                local0 += 1
                dic['nsize'].append(0)
                dic['esize'].append(0)
                dic['is_con'].append(0)
                dic['nsize'][local0] = temp0
                dic['esize'][local0] = temp1
                dic['is_con'][local0] = temp2
            except: print('graph size')
            try:
                self.hist('degree')
                self.hist_log_log('degree')
            except: print('degree hist')
            try:
                temp0, temp1, temp2, temp3 = self.community_detection()
                local1 += 1
                dic['modul'].append(0)
                dic['cover'].append(0)
                dic['modul_r'].append(0)
                dic['cover_r'].append(0)
                dic['modul'][local1], dic['cover'][local1],dic['modul_r'][local1], dic['cover_r'][local1] = temp0, temp1, temp2, temp3
            except: print('community detection')
            try:
                temp0, temp1, temp2, temp3, temp4, temp5 = self.topology_chars()
                local2 += 1
                dic['asph'].append(0)
                dic['cc'].append(0)
                dic['asph_r'].append(0)
                dic['cc_r'].append(0)
                dic['sigma'].append(0)
                dic['omega'].append(0)
                dic['asph'][local2], dic['cc'][local2], dic['asph_r'][local2], dic['cc_r'][local2], dic['sigma'][local2], dic['omega'][local2] = temp0, temp1, temp2, temp3, temp4, temp5
            except: print('topology chars')
            try:
                temp0, attr = self.assortativity(boolean = False)
                local3 += 1
                dic['assort'].append(0)
                dic['assort'][local3] = temp0
            except: print('assortativity')
            try:
                self.graph_correlations()
            except: print('correlations')
            try:
                temp0 = self.rich_club()
                local4 += 1
                dic['rc'].append(0)
                dic['rc_la'].append(0)
                dic['rc'][local4] = temp0
                dic['rc_la'][local4] = '{}'.format(i)
            except: print('rich club')
            try:
                for prop in ['money','asset','approval','worth_ratio','situation']:
                    self.communities_property_dist(prop,boolean=False)
            except: print('property hist')
            try:
                self.intercommunity_links()
            except: print('edge distribution')

        if 'attr' in locals():
            assort = [[] for i in range(len(attr))]
            for i in range(len(attr)):
                for j in np.arange(length):
                    assort[i].append(dic['assort'][j][i])
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
        second_array = list(np.array(dic['is_con'],dtype=int)*self.N)
        self.plot_general(path,dic['nsize'],second_array=second_array,title='GR Number of Nodes in Each Friendship Point')
        self.plot_general(path,dic['esize'],title='GR Number of Edges in Each Friendship Point')
        if len(dic['rc_la']) != 0:
            self.plot_general(path,dic['rc'],indicator=False,label=dic['rc_la'],title='GR Rich Club Vs Friendship Point')
        if 'attr' in locals():
            self.plot_general(path,assort,indicator=False,label=list(attr),title='GR Assortativity Vs Friendship Point')
        
        return dic
    
    pass