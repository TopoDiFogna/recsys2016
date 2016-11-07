import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from datetime import datetime as dt


def apply_shrinkage(self, X, sim):
    #  TODO: compute the shrunk similarity
    # create an "indicator" version of X
    X_ind = X.copy()
    X.ind.data = np.ones_like(X_ind.data)
    # compute the co-rated count
    co_counts = X_ind.T.dot(X_ind).toarray().astype(np.float32)

    # compute the shrinkage value and then multiply it with X
    sim *= co_counts * (co_counts + self.shrinkage)
    return sim


def computeCosine(X, shrinkage):
    # 1) normalize the column of X
    # compute the column-wise norms
    Xsq = X.copy()
    Xsq.data **= 2  # element-+wise square of X
    norm = np.sqrt(Xsq.sum(axis=0))  # matrix[1,M]
    norm = np.asarray(norm).ravel()  # array(M)
    norm += 1e-6

    # compute the number of nonzeroes in each column
    # NOTE: works only if X is csc!!!
    col_nnz = np.diff(X.indptr)
    # then
    # normalize the values in each columns
    X.data /= np.repeat(norm, col_nnz)

    # 2) compute the cosine similarity
    sim = X.T.dot(X).toarray()
    # zero-out the diagonal
    np.fill_diagonal(sim, 0.0)

    if shrinkage > 0.:
        sim = apply_shrinkage(X, sim)

    return sim


items = pd.read_table("data/item_profile.csv", sep="\t")
items.fillna("0", inplace=True)
tagdf = pd.read_csv("tag_matrix.csv", header=0)
title = pd.read_csv("title_matrix.csv", header=None).drop(1, axis=1)

tags = pd.Series(index=tagdf.id, data=np.arange(tagdf.index.size))

itemArray = items.id.values

columnstodrop = ["title", "career_level", "discipline_id", "industry_id", "country", "region", "latitude", "longitude",
                 "employment", "created_at", "active_during_test"]

items.drop(columnstodrop, axis=1, inplace=True)

tagsArray = items.tags.values
columns = []
rows = []
index = 0
print("Computing Sparse COO matrix")
tic = dt.now()
for tagelement in tagsArray:
    if tagelement != "0":
        tagelement = tagelement.split(',')
        for tag in tagelement:
            rows.append(index)
            columns.append(tags.loc[int(tag)])
    index += 1

data = np.ones_like(columns)
tagsparsematrix = coo_matrix((data, (rows, columns)), shape=(itemArray.size, tags.size)).tocsr()
print("Sparse matrix computed in: {}".format(dt.now() - tic))
print("Computing similarities")
tic=dt.now()
sim = computeCosine(tagsparsematrix.tocsc().astype(np.float32), 0)
print("Similarities computed in: {}".format(dt.now() - tic))

