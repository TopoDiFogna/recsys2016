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

def computeNormalization(dataset) :

    # 1) normalize the column of X
    # compute the column-wise norms
    Xsq = dataset.copy()
    Xsq.data **= 2  # element-+wise square of X
    norm = np.sqrt(Xsq.sum(axis=0))  # matrix[1,M]
    norm = np.asarray(norm).ravel()  # array(M)
    norm += 1e-6

    # compute the number of nonzeroes in each column
    # NOTE: works only if X is csc!!!
    col_nnz = np.diff(dataset.indptr)
    # then
    # normalize the values in each columns
    dataset.data /= np.repeat(norm, col_nnz)

    return dataset

def computeCosine(itemIndex, alreadyRatedItems, datasetNormalized, shrinkage):

    slice=datasetNormalized.getcol(itemIndex)

    # 2) compute the cosine similarity
    sim = slice.T.dot(datasetNormalized).toarray()
    # zero-out the diagonal
    zeroToReplace=np.empty_like(itemIndex)
    np.put(sim,itemIndex,zeroToReplace)

    # non credo che sia necessario questo snippet
    # if shrinkage > 0.:
    #     sim = apply_shrinkage(datasetNormalized, sim)

    return sim


##TODO rendere funzione la parte che fa la matrice per una migliore leggibilità del codice
##TODO implementare la funzione per calcolare i ratings dato un poll di user_id
##TODO scrivere la parte di script che calcola tutti gli user_id dei sample e buttarli in pasto alla funzione citata sopra
##TODO scrivere il nuovo file di reccomendations.csv

items = pd.read_table("data/item_profile.csv", sep="\t")
items.fillna("0", inplace=True)
#items=items[items.active_during_test == 1]
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
tagsparsematrix = coo_matrix((data, (rows, columns)), shape=(itemArray.size, tags.size))
print("Sparse matrix computed in: {}".format(dt.now() - tic))
print("Sparse matrix shape {}".format(tagsparsematrix.shape))
print("Computing similarities")
tic=dt.now()
transposedMatrix=tagsparsematrix.T.tocsc()
normalizedMatrix=computeNormalization(transposedMatrix.astype(np.float16))
computeCosine(0, 0,normalizedMatrix,0)
print("Similarities computed in: {}".format(dt.now() - tic))

