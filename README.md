# Inflow-and-Outflow-Centrality

This repository presents implementations of the **inflow** and **outflow** centrality metrics. Additionally it offers functions to compute the **weighted degree** and **closeness centralities**.

These are found in the *utilities.py* file. They take as input the interaction network as a pandas dataframe and a dictionary containing the node ids as keys and the node features as values.

The example folder contains the airport network used in the paper. The interaction network is found in the *AirportNetwork* csv file and the log2 population of each city (features) is saved in the *population* pkl file. The results for inflow and outflow centralities on this network are found in the *inflow_output* and *outflow_output* csv files respectively.

## Python and package versions
*python* == 3.12.4  
*networkx* == 3.6  
*pandas* == 2.2.2  
*numpy* == 1.26.4   

## Publication
```bibtex
@article{papazian2026inflow,
  title={Inflow and outflow centrality: novel centrality metrics inspired by graph convolution},
  author={Papazian, Aram and Helms, Volkhard},
  journal={Applied Network Science},
  volume={11},
  number={1},
  pages={22},
  year={2026},
  publisher={Springer Nature BV}
}
\```
