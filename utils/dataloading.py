import numpy as np
from scipy.sparse import csc_matrix


def load_sparse_csc(filename):
    loader = np.load(filename)
    return csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
