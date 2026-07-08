import pandas as pd
import numpy as np
from collections import Counter,defaultdict
import networkx as nx
import math
import sys
import os
import gzip
from typing import Literal



def flow_metric(interactions: pd.DataFrame, features: pd.DataFrame | None, flow_type: Literal['both', 'in', 'out']):
    """
    flow_metric returns the in/outflow metric for each node in the network

    interactions: pandas dataframe of interactions having 2 columns with node ids.
                  optionally, a third column of float weights can be included;
                  when present, the weights touching each node are summed up
                  (the same way node degrees are) and used to compute a second,
                  weight-based version of the in/outflow metric alongside the
                  original degree-based one.

    features: dictionary with node id as key and respective feature as value.
              make sure features follow somewhat normal distribution. If None,
              features default to 1 for every node, the metric is computed
              once with those defaults, and then recomputed a second time
              using the resulting InflowValue (or WeightedInflowValue, if
              weights are present) as the features. In that case the full
              first-pass results are also included in the output as
              InflowValueFirstPass, OutflowValueFirstPass (and
              WeightedInflowValueFirstPass / WeightedOutflowValueFirstPass if
              weights are present).

    flow_type: specify as either "in", "out", or "both". inflow and outflow are
               always computed together internally; flow_type only controls
               which columns are returned.

    returns pandas dataframe. For flow_type "in"/"out": Node, In/OutflowValue,
    Degree. For flow_type "both": Node, InflowValue, OutflowValue, Degree. If a
    third (float) column was present in interactions, the summed weight per
    node (WeightSum) is added along with the weight-based counterpart(s) of the
    value column(s), e.g. WeightedInflowValue / WeightedOutflowValue, computed
    with degrees replaced by WeightSum. If features was None, the additional
    columns InflowValueFirstPass and OutflowValueFirstPass (and
    WeightedInflowValueFirstPass / WeightedOutflowValueFirstPass, if weights
    are present) hold the first-pass results that were used to derive the
    features for the second pass.
    """

    ############# removing duplicate interactions and self interactions
    duplis=defaultdict(list)
    cols=list(interactions.columns)
    col1=cols[0]
    col2=cols[1]
    has_weights=len(cols)>=3 and pd.api.types.is_float_dtype(interactions[cols[2]])
    col3=cols[2] if has_weights else None
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

    replace_with_result=features is None
    if features is None:
        features={n:1 for n in inters_dictform}

    ####################### get summed edge weight of each node (if provided)
    weightsum=None
    medianweight=None
    if has_weights:
        w1=interactions.groupby(col1)[col3].sum()
        w2=interactions.groupby(col2)[col3].sum()
        weightsum=w1.add(w2,fill_value=0).to_dict()
        medianweight=np.median(list(weightsum.values()))

    ####################### compute in/outflow metric
    mediandegree=np.median(list(degrees.values()))

    def compute_metrics(features):
        inmetric=dict()
        outmetric=dict()
        w_inmetric=dict()
        w_outmetric=dict()
        for i in inters_dictform:
            intot=0
            outtot=0
            w_intot=0
            w_outtot=0
            for j in inters_dictform[i]:
                muldeg=degrees[i]*degrees[j] ## di dj
                sqrt=math.sqrt(muldeg)
                intot+=features[j]/sqrt
                outtot+=features[i]/sqrt
                if has_weights:
                    mulweight=weightsum[i]*weightsum[j]
                    sqrtweight=math.sqrt(mulweight)
                    w_intot+=features[j]/sqrtweight
                    w_outtot+=features[i]/sqrtweight
            denom=math.sqrt(degrees[i])+mediandegree
            inmetric[i]=intot/denom
            outmetric[i]=outtot/denom
            if has_weights:
                denomweight=math.sqrt(weightsum[i])+medianweight
                w_inmetric[i]=w_intot/denomweight
                w_outmetric[i]=w_outtot/denomweight
        return inmetric,outmetric,w_inmetric,w_outmetric

    inmetric,outmetric,w_inmetric,w_outmetric=compute_metrics(features)

    if replace_with_result: ## features were defaulted, refine using first pass result
        fp_inmetric,fp_outmetric,fp_w_inmetric,fp_w_outmetric=inmetric,outmetric,w_inmetric,w_outmetric
        features=fp_w_inmetric if has_weights else fp_inmetric
        inmetric,outmetric,w_inmetric,w_outmetric=compute_metrics(features)

    nodes=list(inmetric.keys())
    result=pd.DataFrame({"Node":nodes,
                         "InflowValue":[inmetric[n] for n in nodes],
                         "OutflowValue":[outmetric[n] for n in nodes],
                         "Degree":[degrees[n] for n in nodes]})
    if has_weights:
        result["WeightSum"]=[weightsum[n] for n in nodes]
        result["WeightedInflowValue"]=[w_inmetric[n] for n in nodes]
        result["WeightedOutflowValue"]=[w_outmetric[n] for n in nodes]
    if replace_with_result:
        result["InflowValueFirstPass"]=[fp_inmetric[n] for n in nodes]
        result["OutflowValueFirstPass"]=[fp_outmetric[n] for n in nodes]
        if has_weights:
            result["WeightedInflowValueFirstPass"]=[fp_w_inmetric[n] for n in nodes]
            result["WeightedOutflowValueFirstPass"]=[fp_w_outmetric[n] for n in nodes]

    if flow_type=="both":
        return result

    elif flow_type=="in":
        keep=["Node","InflowValue","Degree"]
        if has_weights:
            keep+=["WeightSum","WeightedInflowValue"]
        if replace_with_result:
            keep+=["InflowValueFirstPass"]
            if has_weights:
                keep+=["WeightedInflowValueFirstPass"]
        return result[keep]

    else: ## outflow
        keep=["Node","OutflowValue","Degree"]
        if has_weights:
            keep+=["WeightSum","WeightedOutflowValue"]
        if replace_with_result:
            keep+=["OutflowValueFirstPass"]
            if has_weights:
                keep+=["WeightedOutflowValueFirstPass"]
        return result[keep]


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


def read_interactions_csv(csv_path):
    """
    reads an interaction network from a csv file and returns a pandas
    dataframe that can be passed directly as the `interactions` argument to
    flow_metric.

    csv_path: path to a csv file where the first two columns contain the node
              ids of each interaction (see example/AirportNetwork.csv for the
              expected format). An optional third column can hold a numeric
              interaction weight; if present it is cast to float so that
              flow_metric picks it up as edge weights.

    returns pandas dataframe with 2 columns (node ids) or 3 columns (node ids
    and float weights).
    """
    interactions=pd.read_csv(csv_path)
    cols=list(interactions.columns)
    if len(cols)>=3:
        interactions[cols[2]]=interactions[cols[2]].astype(float)
    return interactions


def read_interactions_ea(ea_path):
    """
    reads an interaction network from a .ea file (see example/4efm_intsc.ea
    for the expected format) and returns a pandas dataframe with three
    columns that can be passed directly as the `interactions` argument to
    flow_metric.

    ea_path: path to a .ea file, or a gzip-compressed .ea.gz file (detected by
             the ".gz" suffix and read with the gzip package). Each
             non-header line is expected to be whitespace separated into 4
             fields: <node1> <interaction description> <node2> <score>. Only
             rows whose description contains "combi:all_all" are kept, and
             rows with a negative score are filtered out.

    returns pandas dataframe with columns Node1, Node2, Score.
    """
    opener=gzip.open if str(ea_path).endswith(".gz") else open
    rows=[]
    with opener(ea_path,"rt") as f:
        for line in f:
            parts=line.split()
            if len(parts)!=4:
                continue
            node1,description,node2,score=parts
            if "combi:all_all" not in description:
                continue
            try:
                score=float(score)
            except ValueError:
                continue
            if score<0:
                continue
            rows.append((node1,node2,score))

    return pd.DataFrame(rows,columns=["Node1","Node2","Score"])



def main():
    """
    command line entry point.

    usage: python -m inflow_outflow_centrality.utilities <flow_type> <interactions_file> [features_csv]

    flow_type: "in", "out", or "both", forwarded to flow_metric.
    interactions_file: path to the interaction network file. The file
                       extension determines how it is parsed: ".ea"/".ea.gz"
                       is read with read_interactions_ea, anything else is
                       read with read_interactions_csv.
    features_csv: optional path to a 2 column csv file (node id, feature
                  value) that is turned into the features dict expected by
                  flow_metric. If omitted, every node id found in
                  interactions_file is assigned a feature value of 1.

    writes the resulting dataframe to "<flow_type>_output.csv" in the same
    folder as interactions_file.
    """
    flow_type=sys.argv[1]
    interactions_path=sys.argv[2]

    if interactions_path.endswith(".ea") or interactions_path.endswith(".ea.gz"):
        interactions=read_interactions_ea(interactions_path)
    else:
        interactions=read_interactions_csv(interactions_path)

    if len(sys.argv)>3:
        features_path=sys.argv[3]
        features_df=pd.read_csv(features_path)
        fcols=list(features_df.columns)
        features=dict(zip(features_df[fcols[0]],features_df[fcols[1]]))
    else:
        features = None

    result=flow_metric(interactions,features,flow_type)

    out_dir=os.path.dirname(os.path.abspath(interactions_path))
    out_path=os.path.join(out_dir,f"{flow_type}_output.csv")
    result.to_csv(out_path,index=False)


if __name__=="__main__":
    main()
