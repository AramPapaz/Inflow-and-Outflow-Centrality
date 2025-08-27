import pandas as pd
import numpy as np
from collections import Counter,defaultdict
import networkx as nx
import math
import collections




def flow_metric(interactions,features,flow_type):
    """
    flow_metric returns the in/outflow metric for each node in the network
    
    interactions: pandas dataframe of interactions having 2 columns with node ids
    
    features: dictionary with node id as key and respective feature as value. 
              make sure features follow somewhat normal distribution.

    flow_type: specify as either "in" or "out"

    returns pandas dataframe with 3 columns: first column represents the node ids
                                             second column represents the in/outflow values
                                             third column represents the node degrees
    """

    ############# removing duplicate interactions and self interactions
    duplis=defaultdict(list)
    cols=list(interactions.columns)
    col1=cols[0]
    col2=cols[1]
    rem=[]
    for i in range(interactions.shape[0]): ## duplicates
        id1=interactions.loc[i,col1]
        id2=interactions.loc[i,col2]
        if id1 in duplis:
            if id2 in duplis[id1]:
                rem.append(i)
                continue
            else:
                duplis[id1].append(id2)
        else:
            duplis[id1].append(id2)
        
        if id2 in duplis:
            if id1 in duplis[id2]:
                rem.append(i)
                continue
            else:
                duplis[id2].append(id1)
        else:
            duplis[id2].append(id1)

    interactions=interactions.drop(rem).reset_index(drop=True)

    rem=[]
    for i in range(interactions.shape[0]): ## self interactions
        if interactions.loc[i,col1]==interactions.loc[i,col2]:
            rem.append(i)
    interactions=interactions.drop(i).reset_index(drop=True)

    inters_dictform=defaultdict(list)
    for i in range(interactions.shape[0]):  ## convert dataframe to dict form
        inters_dictform[interactions.loc[i,col1]].append(interactions.loc[i,col2])
        inters_dictform[interactions.loc[i,col2]].append(interactions.loc[i,col1])

    ####################### get degrees of each node
    c1=Counter(interactions[col1])
    c2=Counter(interactions[col2])
    degrees=c1+c2

    ####################### compute in/outflow metric

    if flow_type=="in": ## inflow
        mediandegree=np.median(list(degrees.values()))
        metric=dict()
        for i in inters_dictform:
            tot=0
            for j in inters_dictform[i]:
                muldeg=degrees[i]*degrees[j] ## di dj
                sqrt=math.sqrt(muldeg)
                tot+=features[j]/sqrt
            metric[i]=tot/(math.sqrt(degrees[i])+mediandegree)

        inflow=pd.DataFrame({"Node":list(metric.keys()),"InflowValue":list(metric.values())})
        damount=list()
        for i in range(inflow.shape[0]):
            damount.append(degrees[inflow.loc[i,"Node"]])
        inflow["Degree"]=damount
        return inflow
    
    else: ## outflow
        mediandegree=np.median(list(degrees.values()))
        metric=dict()
        for i in inters_dictform:
            tot=0
            for j in inters_dictform[i]:
                muldeg=degrees[i]*degrees[j] ## di dj
                sqrt=math.sqrt(muldeg)
                tot+=features[i]/sqrt
            metric[i]=tot/(math.sqrt(degrees[i])+mediandegree)

        outflow=pd.DataFrame({"Node":list(metric.keys()),"OutflowValue":list(metric.values())})
        damount=list()
        for i in range(outflow.shape[0]):
            damount.append(degrees[outflow.loc[i,"Node"]])
        outflow["Degree"]=damount
        return outflow
    

def node_weighted_degree_centrality(interactions, features, f=lambda x: x):
    """
    returns node weighted degree centrality

    interactions: pandas dataframe of interactions having 2 columns with node ids
    
    features: dictionary with node id as key and respective feature as value.
    """
    cols=list(interactions.columns)
    col1=cols[0]
    col2=cols[1]
    G=nx.from_pandas_edgelist(interactions, source=col1, target=col2)
    total_weight = sum(f(w) for w in features.values())
    centrality = {}

    for u in G.nodes():
        numerator = sum(f(features[v]) for v in G.neighbors(u))
        centrality[u] = numerator / total_weight

    return centrality



def node_weighted_closeness_centrality(interactions, features, f=lambda x: x):
    """
    returns node weighted closeness centrality

    interactions: pandas dataframe of interactions having 2 columns with node ids
    
    features: dictionary with node id as key and respective feature as value.
    """
    cols=list(interactions.columns)
    col1=cols[0]
    col2=cols[1]
    G=nx.from_pandas_edgelist(interactions, source=col1, target=col2)

    total_weight = sum(f(w) for w in features.values())
    centrality = {}

    for u in G.nodes():
        numerator = f(features[u])
        lengths = nx.single_source_shortest_path_length(G, u)

        for v, d in lengths.items():
            if v != u:
                numerator += f(features[v]) / (d + 1)

        centrality[u] = numerator / total_weight

    return centrality

