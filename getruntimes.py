import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter,defaultdict
import math
import timeit

def flow_metric(interactions,inters_dict,features):
    cols=list(interactions.columns)
    col1=cols[0]
    col2=cols[1]
    
    ####################### get degrees of each node
    c1=Counter(interactions[col1])
    c2=Counter(interactions[col2])
    degrees=c1+c2

    ####################### compute in/outflow metric
    mediandegree=np.median(list(degrees.values()))
    metric=dict()
    for i in inters_dict:
        tot=0
        for j in inters_dict[i]:
            muldeg=degrees[i]*degrees[j] ## di dj
            sqrt=math.sqrt(muldeg)
            tot+=features[j]/sqrt
        metric[i]=tot/(math.sqrt(degrees[i])+mediandegree)
    return metric

def run_im(): ## our metric
    flow_metric(graph,inters_dictform,feats)


def run_ppr(): ## personalized PageRank
    nx.pagerank(G,personalization=feats)

def node_weighted_degree_centrality(G, node_weights, f=lambda x: x):
    total_weight = sum(f(w) for w in node_weights.values())
    centrality = {}

    for u in G.nodes():
        numerator = sum(f(node_weights[v]) for v in G.neighbors(u))
        centrality[u] = numerator / total_weight

    return centrality

def run_wd(): ## Weighted degree centrality
    node_weighted_degree_centrality(G,node_weights=feats)

def node_weighted_closeness_centrality(G, node_weights, f=lambda x: x):
    total_weight = sum(f(w) for w in node_weights.values())
    centrality = {}

    for u in G.nodes():
        numerator = f(node_weights[u])
        lengths = nx.single_source_shortest_path_length(G, u)

        for v, d in lengths.items():
            if v != u:
                numerator += f(node_weights[v]) / (d + 1)

        centrality[u] = numerator / total_weight

    return centrality

def run_wc(): ## weighted closeness centrality
    node_weighted_closeness_centrality(G,node_weights=feats)

def run_ac(): ## alpha centrality
    nx.katz_centrality_numpy(G,beta=feats)
def run_bc(): ## betweenness centrality
    nx.betweenness_centrality(G)

size=[1000,2500,5000,10000]

im=[]
pr=[]
ac=[]
wd=[]
wc=[]
bc=[]

gs=[]

for s in size:
    
    ### generate network
    G = nx.erdos_renyi_graph(s, 0.01)
    graph = pd.DataFrame(G.edges(), columns=["source", "target"])
    
    ### get features
    feats=dict()
    nodes=list(graph['source'])
    nodes.extend(list(graph['target']))
    nodes=list(set(nodes))
    for i in nodes:
        feats[i]=np.random.uniform(1,10)

    ### convert to dictionary
    inters_dictform=defaultdict(list)
    for i in range(graph.shape[0]):  ## convert dataframe to dict form
        inters_dictform[graph.loc[i,'source']].append(graph.loc[i,'target'])
        inters_dictform[graph.loc[i,'target']].append(graph.loc[i,'source'])
    
    im.extend(timeit.repeat(run_im, repeat=10, number=1))
    pr.extend(timeit.repeat(run_ppr, repeat=10, number=1))
    ac.extend(timeit.repeat(run_ac, repeat=10, number=1))
    wd.extend(timeit.repeat(run_wd, repeat=10, number=1))
    wc.extend(timeit.repeat(run_wc, repeat=10, number=1))
    bc.extend(timeit.repeat(run_bc, repeat=10, number=1))

    arr=[s]*10
    gs.extend(arr)


data=pd.DataFrame({"IM & OM":im,"Personalized PageRank":pr,
                   "Alpha Centrality":ac,"Weighted Degree Centrality":wd,
                   "Weighted Closeness Centrality":wc,"Betweenness Centrality":bc,
                   "Number of Nodes":gs})
data.to_csv("RunTimeResults.csv",index=False)

