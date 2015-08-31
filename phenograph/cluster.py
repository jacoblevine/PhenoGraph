import numpy as np
from scipy import sparse as sp
from IPython.parallel import Client, NoEnginesRegistered
from phenograph.core import (gaussian_kernel, parallel_jaccard_kernel, jaccard_kernel,
                             find_neighbors, neighbor_graph, graph2binary, runlouvain)
import subprocess
import time
import re
import os
import uuid


def cluster(data, k=30, directed=False, prune=False, min_cluster_size=10, jaccard=True,
            use_parallel=False, dview=None, shutdown=False):
    """
    PhenoGraph clustering

    :param data: Numpy ndarray of data to cluster
    :param k: Number of nearest neighbors to use in first step of graph construction
    :param directed: Whether to use a symmetric (default) or asymmetric ("directed") graph
        The graph construction process produces a directed graph, which is symmetrized by one of two methods (see below)
    :param prune: Whether to symmetrize by taking the average (prune=False) or produce (prune=True) between the graph
        and its transpose
    :param min_cluster_size: Cells that end up in a cluster smaller than min_cluster_size are considered outliers
        and are assigned to -1 in the cluster labels
    :param jaccard: If True, use Jaccard metric between k-neighborhoods to build graph.
        If False, use a Gaussian kernel.
    :param use_parallel: If True, perform nearest neighbor search in parallel using IPython.
        If True while dview=None (see below), launch a new hub and engines
    :param dview: Instance of IPython.parallel DirectView object for parallel computing.
        Overrides use_parallel setting: provided dview will be used regardless of use_parallel is True or False
    :param shutdown: Whether to shut down dview after the code is executed

    :return communities: numpy integer array of community assignments for each row in data
    :return graph: numpy sparse array of the graph that was used for clustering
    :return Q: the modularity score for communities on graph
    """

    def _sortresult(communities, prune):
        new = np.zeros(communities.shape, dtype=np.int)
        sizes = [sum(communities == x) for x in np.unique(communities)]
        o = np.argsort(sizes)
        o = o[::-1]
        for i, c in enumerate(o):
            if sizes[c] > prune:
                new[communities == c] = i
            else:
                new[communities == c] = -1
        return new

    # NB if prune=True, graph must be undirected, and the prune setting takes precedence
    if prune:
        print("Setting directed=False because prune=True")
        directed = False

    # If direct view is passed, we don't need to go through any of this
    if use_parallel and dview is None:

        kernel = parallel_jaccard_kernel

        # First, check if a cluster is already running
        p = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        p1 = re.compile("IPython.parallel.engine")
        p2 = re.compile("--cluster-id phenograph")
        out = out.decode().split("\n")
        matches = [line for line in out if p1.search(line) and p2.search(line)]
        # If no cluster running, initiate one
        if len(matches) == 0:
            # If we are going to initiate a cluster, also shutdown when finished
            shutdown = True
            # appropriate number of workers depends on system
            n_workers = 8
            print("Launching new cluster with {} workers".format(n_workers), flush=True)
            args = ['ipcluster', 'start', '--n=' + str(n_workers), '--daemonize=True', '--cluster-id=phenograph']
            subprocess.call(args)
            wait = True
            time.sleep(1)
            tries = 0
            while wait and tries < 100:
                try:
                    c = Client(cluster_id="phenograph")
                    dview = c[:]
                    wait = False
                except FileNotFoundError:
                    print("Setting up engines...")
                    tries += 1
                    time.sleep(2)
                except NoEnginesRegistered:
                    print("Waiting for engines to register...")
                    tries += 1
                    time.sleep(2)
            print("Cluster launched successfully", flush=True)
            kernel = parallel_jaccard_kernel
            kernelargs = dict(dview=dview)

        # If a cluster is running but dview wasn't passed, create dview
        else:
            print("Generating direct view of running cluster", flush=True)
            c = Client(cluster_id="phenograph")
            dview = c[:]
            kernelargs = dict(dview=dview)

    # If dview was passed, use this for parallel implementation
    elif dview is not None:
        kernel = parallel_jaccard_kernel
        kernelargs = dict(dview=dview)

    # Otherwise, use serial implementation
    else:
        kernel = jaccard_kernel
        kernelargs = {}  # (idx will be added to kernelargs later)
        if data.shape[0] > 50000:
            print("Computing nearest neighbors on {} cells. Consider using the parallel implementation "
                  "for data of this size.".format(data.shape[0]))

    # Start timer
    tic = time.time()
    # Go!
    d, idx = find_neighbors(data, k=k, dview=dview)
    print("Neighbors computed in {} seconds".format(time.time() - tic), flush=True)
    subtic = time.time()
    kernelargs['idx'] = idx
    # if not using jaccard kernel, use gaussian
    if not jaccard:
        kernelargs['d'] = d
        kernelargs['sigma'] = 1
        kernel = gaussian_kernel
        graph = neighbor_graph(kernel, kernelargs)
        print("Gaussian kernel graph constructed in {} seconds".format(time.time() - subtic), flush=True)
    else:
        graph = neighbor_graph(kernel, kernelargs)
        print("Jaccard graph constructed in {} seconds".format(time.time() - subtic), flush=True)
    if not directed:
        if not prune:
            # symmetrize graph by averaging with transpose
            sg = (graph + graph.transpose()).multiply(.5)
        else:
            # symmetrize graph by multiplying with transpose
            sg = graph.multiply(graph.transpose())
        # retain lower triangle (for efficiency)
        graph = sp.tril(sg, -1)
    # write to file with unique id
    uid = uuid.uuid1().hex
    graph2binary(uid, graph)
    communities, Q = runlouvain(uid)
    print("PhenoGraph complete in {} seconds".format(time.time() - tic), flush=True)
    communities = _sortresult(communities, min_cluster_size)
    # clean up
    for f in os.listdir():
        if re.search(uid, f):
            os.remove(f)

    if shutdown and dview is not None:
        subprocess.call(['ipcluster', 'stop', '--cluster-id=phenograph'])

    return communities, graph, Q
