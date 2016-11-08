import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from datetime import datetime as dt


def apply_shrinkage(self, x, sim):
    # create an "indicator" version of X
    x_ind = x.copy()
    x.ind.data = np.ones_like(x_ind.data)
    # compute the co-rated count
    co_counts = x_ind.T.dot(x_ind).toarray().astype(np.float32)

    # compute the shrinkage value and then multiply it with X
    sim *= co_counts * (co_counts + self.shrinkage)
    return sim


def compute_normalization(dataset):
    # 1) normalize the column of X
    # compute the column-wise norms
    xsq = dataset.copy()
    xsq.data **= 2  # element-+wise square of X
    norm = np.sqrt(xsq.sum(axis=0))  # matrix[1,M]
    norm = np.asarray(norm).ravel()  # array(M)
    norm += 1e-6

    # compute the number of non zeroes in each column
    # NOTE: works only if X is csc!!!
    col_nnz = np.diff(dataset.indptr)
    # then
    # normalize the values in each columns
    dataset.data /= np.repeat(norm, col_nnz)

    return dataset


def compute_cosine(item_index, already_rated_items, dataset_normalized, shrinkage):
    sliced_dataset = dataset_normalized.getcol(item_index)

    # 2) compute the cosine similarity
    sim = sliced_dataset.T.dot(dataset_normalized).toarray()
    # zero-out the diagonal
    zero_to_replace = np.empty_like(item_index)
    np.put(sim, item_index, zero_to_replace)

    # non credo che sia necessario questo snippet
    # if shrinkage > 0.:
    #     sim = apply_shrinkage(datasetNormalized, sim)

    return sim


def createcoomatrix(tags_array):
    print("Computing Sparse COO matrix")
    columns = []
    rows = []
    index = 0
    tic = dt.now()
    for tagelement in tags_array:
        if tagelement != "0":
            tagelement = tagelement.split(',')
            for tag in tagelement:
                rows.append(index)
                columns.append(tags.loc[int(tag)])
        index += 1

    data = np.ones_like(columns)
    print("Sparse matrix computed in: {}".format(dt.now() - tic))
    return coo_matrix((data, (rows, columns)), shape=(itemArray.size, tags.size))


# TODO implementare la funzione per calcolare i ratings dato un poll di user_id
# TODO scrivere la parte di script che calcola tutti gli user_id dei sample e buttarli in pasto alla funzione citata sopra
# TODO scrivere il nuovo file di reccomendations.csv

items = pd.read_table("data/item_profile.csv", sep="\t")
items.fillna("0", inplace=True)
# items=items[items.active_during_test == 1] SERVE?
tagdf = pd.read_csv("tag_matrix.csv", header=0)
title = pd.read_csv("title_matrix.csv", header=None).drop(1, axis=1)

tags = pd.Series(index=tagdf.id, data=np.arange(tagdf.index.size))

itemArray = items.id.values

columnstodrop = ["title", "career_level", "discipline_id", "industry_id", "country", "region", "latitude", "longitude",
                 "employment", "created_at", "active_during_test"]
items.drop(columnstodrop, axis=1, inplace=True)

# Computing the COO matrix
tagsparsematrix = createcoomatrix(items.tags.values)
print("Sparse matrix shape {}".format(tagsparsematrix.shape))


# print("Computing similarities")
# tic = dt.now()
# transposedMatrix=tagsparsematrix.T.tocsc()
# normalizedMatrix=computeNormalization(transposedMatrix.astype(np.float16))
# computeCosine(0, 0, normalizedMatrix, 0)
# print("Similarities computed in: {}".format(dt.now() - tic))
