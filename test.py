import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx
from sys import maxsize

filename = './data/test'
filename = './data/houses_to_rent_v2'
filename = './data/winequalityN'

from tqdm import tqdm

from os import listdir

def distance(a,b):
    _ = sum([(a[i]-b[i])**2 for i in a.index])
    return (_)**0.5

# def detect_outlier(filename, on_cluster=True, plot=False):
def detect_outlier(df, on_cluster=True, plot=False):
    # CREATING CLUSTERS WITH K MEANS
    # df = pd.read_csv(filename+'.csv')
    df = df.dropna()
    df = df.select_dtypes([np.number])
    k = int((len(df))**0.5)
    kmeans_result = KMeans(k).fit(df)
    columns = df.columns
    df['class'] = kmeans_result.labels_
    
    # CREATING DATAFRAME OF CLUSTER
    cluster = pd.DataFrame(kmeans_result.cluster_centers_,columns=columns)
    count = [len(df[df['class']==label]) for label in list(set(df['class']))]
    
    k = int(len(cluster)**0.5) if on_cluster else len(set(df))
    current_df = cluster if on_cluster else df
    length = len(current_df)
    
    if (length>k):
        
        # CALCULATING ALL THE DISTANCES 
        kruskal_input = []
        for i in tqdm(range (length),desc="Loading ..."):
            for j in range (i+1,length):
                kruskal_input.append((i,j,distance(current_df.iloc[i],current_df.iloc[j])))
        kruskal_input.sort()
        
        # MAKING THE MST
        from mst import Graph as mst_graph
        M = mst_graph(length)
        for edge in kruskal_input:
            M.addEdge(edge[0],edge[1],edge[2])
        M.KruskalMST()
        # MST AVAILABLE IN M.result
        
        # DFS TO DETECT DISCONNECTED COMPONENTS
        from dfs import Graph as dfs_graph
        D = dfs_graph()
        dfs_edges = [(_[2],_[0],_[1]) for _ in M.result]
        dfs_edges.sort(reverse=True)
        removed = dfs_edges[:k]
        dfs_edges = dfs_edges[k:]
        for edge in dfs_edges:
            D.add_edge(edge[1],edge[2])
            D.add_edge(edge[2],edge[1])
        disconnected = {}
        for edge in removed:
            # DFS RESULT AVAILABLE IN D.temp
            D.dfs(edge[1])
            disconnected[edge[1]] = D.temp
            D.dfs(edge[2])
            disconnected[edge[2]] = D.temp
        
        # DETECTING OUTLIERS
        outlier = []
        minimum_size = maxsize 
        for key in disconnected.keys():
            minimum_size = min(minimum_size,len(disconnected[key]))
        for key in disconnected.keys():
            if len(disconnected[key])==minimum_size:
                outlier = outlier+list(disconnected[key])
        outlier =list(set(outlier))
        current_df['outlier'] = [True if i in outlier else False for i in range(len(current_df))]
        current_df['vertex'] = [i for i in range(len(current_df))]
        if on_cluster:
            current_df['count'] = count
        
        # print (current_df.head(20))
        # percentage = (sum(current_df[current_df['outlier']==True]['count'])*100)/sum(current_df['count']) if on_cluster else (len(outlier)*100)/len(current_df)
        # print (percentage)
        
        # PLOTTING RESULTS
        non_outlier = []
        for i in range (len(current_df)):
            if i not in outlier:
                non_outlier.append(i)
        if (plot) :
            P = nx.Graph()
            for edge in M.result:
                P.add_node(edge[0])
                P.add_node(edge[1])
                # P.add_edge(edge[0],edge[1])
            for edge in dfs_edges:
                # P.add_node(edge[1])
                # P.add_node(edge[2])
                P.add_edge(edge[1],edge[2])
            nx.draw(P, with_labels=True)
            plt.show()
        return current_df

# BELOW FOR TESTING ONLY 
if __name__ == "__main__":

    if filename+'_fitted.csv' not in listdir(): 
        df = pd.read_csv(filename+'.csv')
        df = df.dropna()
        df = df.select_dtypes([np.number])
        k = int((len(df))**0.5)
        kmeans_result = KMeans(k).fit(df)
        columns = df.columns
        df['class'] = kmeans_result.labels_
        cluster = pd.DataFrame(kmeans_result.cluster_centers_,columns=columns)
        cluster['count'] = [len(df[df['class']==label]) for label in list(set(df['class']))]
        df.to_csv(filename+'_fitted.csv',index=False)
        cluster.to_csv(filename+'_cluster.csv',index=False)

    else : 
        on_cluster = True

        df = pd.read_csv(filename+'_fitted.csv')
        columns = list(df.columns)
        columns.remove('class')
        df_cluster = pd.read_csv(filename+'_cluster.csv')
        count = df_cluster['count']
        df_cluster = df_cluster[columns]
        k = int(len(df_cluster)**0.5) if on_cluster else len(set(df))

        current_df = df_cluster if on_cluster else df
        kruskal_input = []
        length = len(current_df)

        if (length>k):
            # calulating all the distances 
            for i in tqdm(range (length),desc="Loading ..."):
                for j in range (i+1,length):
                    kruskal_input.append((i,j,distance(current_df.iloc[i],current_df.iloc[j])))
            kruskal_input.sort()

            # MAKING THE MST
            from mst import Graph as mst_graph
            M = mst_graph(length)
            for edge in kruskal_input:
                M.addEdge(edge[0],edge[1],edge[2])
            M.KruskalMST()
            # MST AVAILABLE IN M.result

            # DOING DFS
            from dfs import Graph as dfs_graph
            D = dfs_graph()
            dfs_edges = [(_[2],_[0],_[1]) for _ in M.result]
            dfs_edges.sort(reverse=True)
            removed = dfs_edges[:k]
            dfs_edges = dfs_edges[k:]
            for edge in dfs_edges:
                D.add_edge(edge[1],edge[2])
                D.add_edge(edge[2],edge[1])
            disconnected = {}
            for edge in removed:
                # DFS RESULT AVAILABLE IN D.temp
                D.dfs(edge[1])
                disconnected[edge[1]] = D.temp
                D.dfs(edge[2])
                disconnected[edge[2]] = D.temp

            # DETECTING OUTLIERS
            outlier = []
            minimum_size = maxsize 
            for key in disconnected.keys():
                minimum_size = min(minimum_size,len(disconnected[key]))
            for key in disconnected.keys():
                if len(disconnected[key])==minimum_size:
                    outlier = outlier+list(disconnected[key])
            outlier =list(set(outlier))

            current_df['outlier'] = [True if i in outlier else False for i in range(len(current_df))]
            current_df['vertex'] = [i for i in range(len(current_df))]
            if on_cluster:
                current_df['count'] = count
            print (current_df.head(20))
            percentage = (sum(current_df[current_df['outlier']==True]['count'])*100)/sum(current_df['count']) if on_cluster else (len(outlier)*100)/len(current_df)
            print (percentage)

            # PLOTTING RESULTS
            P = nx.Graph()
            color_map_density = []
            color_map = []
            
            # ADDING ALL VERTICES FROM MST
            for edge in M.result:
                if edge[0] not in list(P.nodes):
                    P.add_node(edge[0])
                    if edge[0] in outlier : color_map.append('gray')
                    else : color_map.append('gold')
                    color_map_density.append(length-current_df.iloc[edge[0]]['count'])
                if edge[1] not in list(P.nodes):
                    P.add_node(edge[1])
                    if edge[1] in outlier : color_map.append('gray')
                    else : color_map.append('gold')
                    color_map_density.append(length-current_df.iloc[edge[1]]['count'])
                # P.add_edge(edge[0],edge[1])
            
            # ADDING EDGES FROM DFS ONLY
            for edge in dfs_edges:
                # P.add_node(edge[1])
                # P.add_node(edge[2])
                P.add_edge(edge[1],edge[2])

            # nx.draw(P, with_labels=True)
            nx.draw(P, with_labels=True,node_color=color_map)
            # nx.draw(P, with_labels=True,node_color=range((length)))
            # nx.draw(P, with_labels=True,node_color=color_map_density)
            
            plt.suptitle("Outliers in %s dataset"%filename)
            plt.show()