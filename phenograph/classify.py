import numpy as np
from phenograph.core import find_neighbors, neighbor_graph, jaccard_kernel
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def random_walk_probabilities(A, labels):

    if labels.min() != 0:
        raise ValueError("Labels should encode unlabeled points with 0s")

    # Check if input is sparse
    if sp.issparse(A):
        if not isinstance(A, sp.csr_matrix):
            A = A.tocsr()
        D = sp.diags(A.sum(axis=1).flatten(), [0], shape=A.shape, format='csr')
        L = D - A
        seeds = labels != 0
        Lu = L[np.invert(seeds), :]  # unlabeled rows
        Lu = Lu.tocsc()[:, np.invert(seeds)]  # unlabeled columns
        # Check that Lu has the right size
        if not all([t == sum(np.invert(seeds)) for t in Lu.shape]):
            raise IndexError("Lu should be square and match size of test data")
        BT = L[np.invert(seeds), :]  # unlabeled rows
        BT = BT.tocsc()[:, seeds]  # labeled columns
        if not (sum(np.invert(seeds)), sum(seeds)) == BT.shape:
            raise IndexError("BT size is incorrect")
        # Matrix representation of labels
        i, j, s = [], [], []
        classes = np.unique(labels[seeds.nonzero()[0]])
        for k in classes:
            i.extend(np.where(labels[seeds] == k)[0])
            j.extend(np.tile(k, sum(seeds == k)))
            s.extend(np.tile(1, sum(seeds == k)))
        i = np.arange(seeds.sum())
        j = labels[seeds] - 1
        s = np.ones((seeds.sum(), ))
        M = sp.coo_matrix((s, (i, j)), shape=(seeds.sum(), len(classes))).tocsc()
        # P = sp.linalg.spsolve(Lu.tocsc(), -BT.dot(M))
        # Use iterative solver
        B = -BT.dot(M)
        vals = [sp.linalg.isolve.bicgstab(Lu, b.T.todense()) for b in B.T]
        warnings = [x[1] for x in vals]
        if any(warnings):
            print("Warning: iterative solver failed to converge in at least one case", flush=True)
        P = normalize(np.vstack(tuple((x[0] for x in vals))).T, norm='l1')

    else:
        D = np.diag(np.sum(A, axis=1))
        L = D - A  # graph laplacian
        seeds = np.array([e != 0 for e in labels], dtype=bool)
        Lu = L[np.invert(seeds), np.invert(seeds)]  # unlabeled rows, unlabeled cols
        BT = L[np.invert(seeds), seeds]  # unlabeled rows, labeled cols
        classes = np.unique(labels[labels > 0])
        M = np.zeros((seeds.sum(), len(classes)))
        for k in classes:
            M[labels[seeds] == k, k] = 1
        P = np.linalg.lstsq(Lu, np.dot(-BT, M))[0]

    return P


def create_graph(data, k=30, metric='euclidean', n_jobs=-1):
    # def _kernel(dxy, sigma=1):
    #     return np.exp(-dxy ** 2 / sigma)

    _, idx = find_neighbors(data, k=k, metric=metric, n_jobs=n_jobs)
    # affinities = np.apply_along_axis(lambda x: _kernel(x, x.std()), axis=1, arr=d)
    # n, k = idx.shape
    # i = [np.tile(x, (k, )) for x in range(n)]
    # i = np.concatenate(np.array(i))
    # j = np.concatenate(idx)
    # s = np.concatenate(affinities)
    # graph = sp.coo_matrix((s, (i, j)), shape=(n, n)).tocsr()
    # graph = (graph + graph.transpose()).multiply(.5)
    graph = neighbor_graph(jaccard_kernel, {'idx': idx})
    # make symmetric
    # graph = (graph + graph.transpose()).multiply(.5)
    return graph


def preprocess(train, test):
    labels = np.zeros((test.shape[0], ), dtype=int)
    data = test
    for c, examples in enumerate(train):
        labels = np.append(labels, np.tile(c+1, (examples.shape[0], )), axis=0)
        data = np.append(data, examples, axis=0)
    # Check that results are valid
    if labels[-1] == 0:
        raise IndexError("Last entry in labels should not be 0")
    if labels.shape[0] != data.shape[0]:
        raise IndexError("Data and labels should be the same length")
    if sum(labels == 0) != test.shape[0]:
        raise IndexError("Labels should include one 0 for every row of test data")
    return data, labels


def classify(train, test, k=30, metric='euclidean', n_jobs=-1):
    """
    Semi-supervised classification by random walks on a graph
    :param train: list of numpy arrays. Each array has a row for each class observation
    :param test: numpy array of unclassified data
    :return c: class assignment for each row in test
    :return P: class probabilities for each row in test
    """
    data, labels = preprocess(train, test)
    # Build graph
    A = create_graph(data, k, metric=metric, n_jobs=n_jobs)
    P = random_walk_probabilities(A, labels)
    c = np.argmax(P, axis=1)
    return c, P

