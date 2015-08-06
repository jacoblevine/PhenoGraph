import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import sparse as sp
import subprocess
import time
import re
import os
import platform
from pkg_resources import Requirement, resource_filename


def find_neighbors(data, k=30, dview=None, metric='euclidean'):
    """
    Wraps sklearn.neighbors.NearestNeighbors, supporting IPython parallel usage

    :param data: n-by-d data matrix
    :param k: number for nearest neighbors search
    :param dview: direct view of ipcluster, if using parallel computation
    :return d: n-by-k matrix of distances
    :return idx: n-by-k matrix of neighbor indices
    """
    if metric == 'cosine' or metric == 'correlation':
        algorithm = 'brute'
    else:
        algorithm = 'ball_tree'

    if dview is not None:

        # Build ball tree
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm=algorithm).fit(data)
        # Send data to workers
        dview.push(dict(nbrs=nbrs))
        dview.scatter('data', data)
        # Run nearest-neighbor search on workers
        dview.execute("d, idx = nbrs.kneighbors(data)")
        # Gather results
        idx = np.array(list(dview.gather('idx')))
        d = np.array(list(dview.gather('d')))

    else:
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm=algorithm).fit(data)
        d, idx = nbrs.kneighbors(data)

    # remove self-distance
    idx = np.delete(idx, 0, axis=1)
    d = np.delete(d, 0, axis=1)
    return d, idx


def neighbor_graph(kernel, kernelargs):
    """
    Apply kernel (i.e. affinity function) to kernelargs (containing information about the data)
    and return graph as a sparse COO matrix

    :param kernel: affinity function
    :param kernelargs: dictionary of keyword arguments for kernel
    :return graph: n-by-n COO sparse matrix
    """
    i, j, s = kernel(**kernelargs)
    n, k = kernelargs['idx'].shape
    graph = sp.coo_matrix((s, (i, j)), shape=(n, n))
    return graph


def gaussian_kernel(idx, d, sigma):
    """
    For truncated list of k-nearest distances, apply Gaussian kernel
    Assume distances in d are Euclidean
    :param idx:
    :param d:
    :param sigma:
    :return:
    """
    n, k = idx.shape
    i = [np.tile(x, (k,)) for x in range(n)]
    i = np.concatenate(np.array(i))
    j = np.concatenate(idx)
    d = np.concatenate(d)
    f = np.vectorize(lambda x: 1/(sigma * (2 * np.pi) ** .5) * np.exp(-.5 * (x / sigma) ** 2))
    # apply vectorized gaussian function
    p = f(d)
    return i, j, p


def jaccard_kernel(idx):
    """
    Compute jaccard coefficient between nearest-neighbor sets
    :param idx: numpy array of nearest-neighbor indices
    :return (i, j, s): tuple of indices and jaccard coefficients, suitable for constructing COO matrix
    """
    n, k = idx.shape
    i = [np.tile(x, (k, )) for x in range(n)]
    i = np.concatenate(np.array(i))
    j = np.concatenate(idx)
    s = list()
    for q in range(n):
        sharedneighbors = [len(np.intersect1d(idx[q], hood)) for hood in idx[idx[q]]]
        s.extend([w / (2 * k - w) for w in sharedneighbors])
    return i, j, s


def parallel_jaccard_kernel(idx, dview):
    """
    Same as jaccard_kernel but accepts an IPython parallel dview instance
    to perform the computations in parallel on multiple cores
    :param idx:
    :param dview:
    :return (i, j, s):
    """
    def calc_jaccard(idx, x):
        s = []
        _, k = idx.shape
        for i in x:
            sharedneighbors = [len(np.intersect1d(idx[i], hood)) for hood in idx[idx[i]]]
            s.extend([w / (2 * k - w) for w in sharedneighbors])
        return s

    # In case there is no active cluster, run serial version
    if dview is not None:

        n, k = idx.shape
        i = [np.tile(x, (k, )) for x in range(n)]
        i = np.concatenate(np.array(i))
        j = np.concatenate(idx)

        dview.push(dict(calc_jaccard=calc_jaccard, idx=idx))
        dview.scatter('x', np.arange(n))
        dview.execute("s = calc_jaccard(idx, x)")
        s = list(dview.gather('s'))

    else:

        i, j, s = jaccard_kernel(idx)

    return i, j, s


def graph2binary(filename, graph):
    """
    Write (weighted) graph to binary file filename.bin
    :param filename:
    :param graph:
    :return None: graph is written to filename.bin
    """
    tic = time.time()
    # Unpack values in graph
    i, j = graph.nonzero()
    s = graph.data
    # place i and j in single array as edge list
    ij = np.hstack((i[:, np.newaxis], j[:, np.newaxis]))
    # add dummy self-edges for vertices at the END of the list with no neighbors
    ijmax = np.union1d(i, j).max()
    n = graph.shape[0]
    missing = np.arange(ijmax+1, n)
    for q in missing:
        ij = np.append(ij, [[q, q]], axis=0)
        s = np.append(s, [0.], axis=0)
    # Check data types: int32 for indices, float64 for weights
    if ij.dtype != np.int32:
        ij = ij.astype('int32')
    if s.dtype != np.float64:
        s = s.astype('float64')
    # write to file (NB f.writelines is ~10x faster than np.tofile(f))
    with open(filename + '.bin', 'w+b') as f:
        f.writelines([e for t in zip(ij, s) for e in t])
    print("Wrote graph to binary file in {} seconds".format(time.time() - tic))


def runlouvain(filename):
    """
    From binary graph file filename.bin, optimize modularity by running multiple random re-starts of
    the Louvain C++ code.

    Louvain is run repeatedly until modularity has not increased in some number (20) of runs
    or if the total number of runs exceeds some larger number (100)

    :param filename:
    :return communities:
    :return Q:
    """
    def get_modularity(msg):
        pattern = re.compile('modularity increased from -*0.\d+ to 0.\d+')
        matches = pattern.findall(msg.decode())
        q = list()
        for line in matches:
            q.append(line.split(sep=" ")[-1])
        return list(map(float, q))

    print('Running Louvain modularity optimization', flush=True)
    
    # Use package location to find Louvain code
    lpath = os.path.abspath(resource_filename(Requirement.parse("PhenoGraph"), 'louvain'))
    try:
        assert os.path.isdir(lpath)
    except AssertionError:
        print("Could not find Louvain code, tried: {}".format(lpath), flush=True)

    # Determine if we're using Windows or Mac/Linux
    if platform.system() == "Windows":
        convert_binary = "\convert.exe"
        community_binary = "\community.exe"
        hierarchy_binary = "\hierarchy.exe"
    else:
        convert_binary = "/convert"
        community_binary = "/community"
        hierarchy_binary = "/hierarchy"
    
    tic = time.time()

    # run convert
    args = [lpath + convert_binary, '-i', filename + '.bin', '-o',
            filename + '_graph.bin', '-w', filename + '_graph.weights']
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    # check for errors from convert
    if bool(out) or bool(err):
        print("stdout from convert: {}".format(out.decode()))
        print("stderr from convert: {}".format(err.decode()))

    Q = 0
    run = 0
    updated = 0
    while run - updated < 20 and run < 100:

        # run community
        fout = open(filename + '.tree', 'w')
        args = [lpath + community_binary, filename + '_graph.bin', '-l', '-1', '-v', '-w', filename + '_graph.weights']
        p = subprocess.Popen(args, stdout=fout, stderr=subprocess.PIPE)
        # Here, we print communities to filename.tree and retain the modularity scores reported piped to stderr
        _, msg = p.communicate()
        fout.close()
        # get modularity from err msg
        q = get_modularity(msg)
        run += 1

        # continue only if we've reached a higher modularity than before
        if q[-1] > Q:

            Q = q[-1]
            updated = run

            # run hierarchy
            args = [lpath + hierarchy_binary, filename + '.tree']
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            # find number of levels in hierarchy and number of nodes in graph
            nlevels = int(re.findall('\d+', out.decode())[0])
            nnodes = int(re.findall('level 0: \d+', out.decode())[0].split(sep=" ")[-1])

            # get community assignments at each level in hierarchy
            hierarchy = np.empty((nnodes, nlevels), dtype='int')
            for level in range(nlevels):
                    args = [lpath + hierarchy_binary, filename + '.tree', '-l', str(level)]
                    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = p.communicate()
                    h = np.empty((nnodes,))
                    for i, line in enumerate(out.decode().splitlines()):
                        h[i] = int(line.split(sep=' ')[-1])
                    hierarchy[:, level] = h

            communities = hierarchy[:, nlevels-1]

            print("After {} runs, maximum modularity is Q = {}".format(run, Q), flush=True)

    print("Louvain completed {} runs in {} seconds".format(run, time.time() - tic), flush=True)

    return communities, Q
