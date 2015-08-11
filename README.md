Welcome to PhenoGraph for Python!
================================

PhenoGraph is a clustering method designed for high-dimensional single-cell data. It works by creating a graph 
("network") representing phenotypic similarities between cells and then identifying communities in this graph.

Included in this software package are compiled binaries that run community detection based on C++ code written by
E. Lefebvre and J.-L. Guillaume in 2008 (["Louvain method"](https://sites.google.com/site/findcommunities/)). The code
has been altered to interface more efficiently with the Python code here. It should work on reasonably new Mac and
Windows machines.

To install PhenoGraph, simply run the setup script:

```
python setup.py install
```

Currently, expected use is during an interactive terminal session (i.e., we do not include command-line support).
The data are expected to be passed as a numpy ndarray.

Support is provided to take advantage of IPython's parallel computing packages, which greatly enhances speed for large
data sets.
 
To run basic clustering:

```
import phenograph
communities, graph, Q = phenograph.cluster(data)
```
For a dataset of $N$ rows, `communities` will be a length $N$ vector of integers specifying a community assignment for each row
in the data. Any rows assigned $-1$ were identified as *outliers* and should not be considered as a member of any community.
`graph` is a $N x N$ scipy.sparse (adjacency) matrix representing the graph used for community detection. `Q` is the modularity score for `communities` as applied to `graph`.
