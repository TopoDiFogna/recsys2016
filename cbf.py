import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from datetime import datetime as dt

# Loading Data
items = pd.read_table("data/item_profile.csv", sep="\t")
tagdf = pd.read_csv("tag_matrix.csv", header=0)
title = pd.read_csv("title_matrix.csv", header=None).drop(1, axis=1)
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
# End loading data

# Data pre processing
items.fillna("0", inplace=True)
itemArray = items.id.values
columnstodrop = ["title", "career_level", "discipline_id", "industry_id", "country", "region", "latitude", "longitude",
                 "employment", "created_at", "active_during_test"]
items.drop(columnstodrop, axis=1, inplace=True)

tags = pd.Series(index=tagdf.id, data=np.arange(tagdf.index.size))

filteredItems = items[items.active_during_test == 1].reset_index()


# End of data pre processing

# Functions
def apply_shrinkage(shrinkage, x, sim):
    # create an "indicator" version of X
    x_ind = x.copy()
    x.ind.data = np.ones_like(x_ind.data)
    # compute the co-rated count
    co_counts = x_ind.T.dot(x_ind).toarray().astype(np.float32)

    # compute the shrinkage value and then multiply it with X
    sim *= co_counts * (co_counts + shrinkage)
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
    if shrinkage > 0.:
        sim = apply_shrinkage(shrinkage, dataset_normalized, sim)

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


def getuserratings(userid):
    sampleinteractions = interactions.loc[interactions['user_id'] == userid].reset_index().drop("index", 1).drop(
        "created_at", 1)
    sampleinteractions = sampleinteractions.groupby(by='item_id', as_index=False).apply(
        lambda x: x.ix[x.interaction_type.idxmax()])
    sample_items = items[items["id"].isin(sampleinteractions.item_id.values)].reset_index()
    inversesampleitems = filteredItems[~filteredItems['id'].isin(sampleinteractions.item_id.values)].reset_index()
    return [sample_items, inversesampleitems]


def compute_ratings(sim_matrix):    # sim_matrix: all items x rated items
    numerator = 0                   # TODO da controllare i calcoli e i vettori dei for
    denominator = 0
    ratings = []
    for rated_item in sim_matrix.columns.values:
        for item in sim_matrix.rated_item.values:
            numerator = numerator + interactions[interactions.item_id == rated_item].interaction_type * sim_matrix.item
            denominator = denominator + item
        ratings.append(numerator/denominator)
    return ratings

# TODO scrivere la parte di script che calcola tutti gli user_id dei sample e buttarli in pasto
# TODO alla funzione citata sopra
# TODO scrivere il nuovo file di reccomendations.csv

# Computing the COO matrix
tagsparsematrix = createcoomatrix(items.tags.values)
print("Sparse matrix shape {}".format(tagsparsematrix.shape))

# Start computing similarities
print("Computing similarities")
tic = dt.now()
transposedMatrix = tagsparsematrix.T.tocsc()
normalizedMatrix = compute_normalization(transposedMatrix.astype(np.float16))
compute_cosine(0, 0, normalizedMatrix, 0)
print("Similarities computed in: {}".format(dt.now() - tic))
# End of computing similarities
