import numpy as np
import pandas as pd
import re
from scipy.sparse import coo_matrix, hstack
from datetime import datetime as dt

# Loading Data
items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
tagdf = pd.read_csv("tag_matrix.csv", header=0)
title = pd.read_csv("title_matrix.csv", header=None).drop(1, axis=1)
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
samples = pd.read_csv("data/sample_submission.csv", header=0)
# End loading data

# Data pre processing
items.fillna("0", inplace=True)
items = items.sort(columns="id", axis=0).reset_index()
itemArray = items.id.values
filteredItems = items[items.active_during_test == 0].id.values
filteredItemsIndexes = items[items.active_during_test == 0].index
columnstodrop = ["title", "career_level", "discipline_id", "industry_id", "country", "region", "latitude", "longitude",
                 "employment", "created_at", "active_during_test"]
items_dropped = items.drop(columnstodrop, axis=1, inplace=False)
tags = pd.Series(index=tagdf.id, data=np.arange(tagdf.index.size))
samplesIds = samples.user_id.values


# End of data pre processing


# Functions
def getIndexOfItem(item_id):
    index = items[items["id"] == item_id].index.tolist()[0]
    return index


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
    zero_to_replace = np.empty_like(already_rated_items)
    np.put(sim, already_rated_items, zero_to_replace)

    # non credo che sia necessario questo snippet
    if shrinkage > 0.:
        sim = apply_shrinkage(shrinkage, dataset_normalized, sim)

    return coo_matrix(sim).T


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
    sampleRatings = sampleinteractions.interaction_type.values
    return sampleRatings


def getuseritems(userid):
    sampleinteractions = interactions.loc[interactions['user_id'] == userid].reset_index().drop("index", 1).drop(
        "created_at", 1)
    if sampleinteractions.empty:
        return []
    else:
        sampleinteractions = sampleinteractions.groupby(by='item_id', as_index=False).apply(
            lambda x: x.ix[x.interaction_type.idxmax()])
        sampleItems = sampleinteractions.item_id.values
        return sampleItems.tolist()
    return 0


def compute_ratings(sim_matrix, sampleRating):  # sim_matrix: all items x rated items

    rated_sim = sim_matrix.copy()
    rated_sim = rated_sim.multiply(sampleRating)
    numerator = np.array(rated_sim.sum(axis=1)).flatten()
    denominator = np.array(sim_matrix.sum(axis=1)).flatten()
    return numerator / denominator


def getitemsid(item_indexes):
    return items.loc[item_indexes].id.values


# TODO scrivere il nuovo file di reccomendations.csv

# Computing the COO matrix
tagsparsematrix = createcoomatrix(items_dropped.tags.values)
print("Sparse matrix shape {}".format(tagsparsematrix.shape))

# Start computing similarities
print("Computing similarities")
transposedMatrix = tagsparsematrix.T.tocsc()
normalizedMatrix = compute_normalization(transposedMatrix.astype(np.float16))

restult = pd.DataFrame(index=np.arange(samplesIds.size), columns=samples.columns.values)

# Start computing ratings for every sample_id
columns = ['user_id', 'recommended_items']
df = pd.DataFrame(index=range(10000), columns=columns)
zero_to_replace = np.empty_like(filteredItems)
index = 0
totaltic = dt.now()
for user_id in samplesIds:
    tic = dt.now()
    items_Rated = getuseritems(user_id)
    items_reccomended = []
    if len(items_Rated) == 0:
        items_reccomended = np.array([1053452, 2778525, 1244196, 1386412, 657183])
    else:
        rated_ids = list(map(getIndexOfItem, items_Rated))
        sim = compute_cosine(rated_ids[0], rated_ids, normalizedMatrix, 0)
        rated_ids_new = list(rated_ids)
        rated_ids.pop(0)
        for id in rated_ids:
            new_sim_col = compute_cosine(id, rated_ids_new, normalizedMatrix, 0)
            sim = hstack([sim, new_sim_col])
        ratings = compute_ratings(sim, getuserratings(user_id))
        np.put(ratings, filteredItemsIndexes, zero_to_replace)
        top_rated_items_id = ratings.argsort()[-5:][::-1]
        items_reccomended = getitemsid(top_rated_items_id)
    df.loc[index] = [user_id, re.sub('[\[\]]', '', np.array_str(items_reccomended))]
    print("Similarities for {} one user computed in: {}".format(index, dt.now() - tic))
    index += 1
print("All similarities computed in {}".format(dt.now() - totaltic))
# End of computing similarities

with open("recommendations_new.csv", "w") as f:
    df.to_csv(f, sep=',', index=False)
