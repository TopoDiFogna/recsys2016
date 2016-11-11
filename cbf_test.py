import numpy as np
import pandas as pd
from scipy import sparse
from datetime import datetime as dt
from sklearn.metrics.pairwise import cosine_similarity

# Loading Data
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
samples = pd.read_csv("data/sample_submission.csv", header=0)
items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
tagdf = pd.read_csv("tag_matrix.csv", header=0)
# End loading data

# Data pre processing
samplesIds = samples.user_id.values  # sample users ids
tags = pd.Series(index=tagdf.id, data=np.arange(tagdf.index.size))  # containing all tags as index
items.fillna("0", inplace=True)
itemArray = items.id.values  # list of items ids
# End of data pre processing


def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse_csc(filename):
    loader = np.load(filename)
    return sparse.csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


# Gets the ratings a user has performed dropping the duplicates and keeping the highest
def getuserratings(userid):
    sampleinteractions = interactions.loc[interactions['user_id'] == userid].reset_index().drop("index", 1).drop(
        "created_at", 1).drop("user_id", 1)
    sampleinteractions = sampleinteractions.groupby(by='item_id', as_index=False).apply(
        lambda x: x.ix[x.interaction_type.idxmax()])
    if sampleinteractions.empty:
        return np.array([])
    else:
        return sampleinteractions.item_id.values


# Creates a sparse matrix where rows are items and cols are tags.
# Inside the matrix there's 1 if the item has the tag, 0 otherwise
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
    return sparse.coo_matrix((data, (rows, columns)), shape=(itemArray.size, tags.size))


def compute_similarities(sparse_matrix):
    print("Computing Similarities")
    tic = dt.now()
    numrows, numcols = sparse_matrix.shape
    similarities = sparse.lil_matrix((numrows, numrows))
    for row_index in range(numrows):
        for col_index in range(numcols):
            similarities[row_index, col_index] = cosine_similarity(sparse_matrix.getrow(row_index),
                                                                   sparse_matrix.getrow(col_index))
        print("finito item {}".format(row_index))
    print("Similarities computed in {}".format(dt.now() - tic))
    return similarities


def compute_item_similarities(index, sparse_matrix):
    print("Computing Similarities")
    tic = dt.now()
    numrows, numcols = sparse_matrix.shape
    similarities = sparse.lil_matrix((1, numrows))
    for col_index in range(numcols):
        similarities[0, col_index] = cosine_similarity(sparse_matrix.getrow(index), sparse_matrix.getrow(col_index))
    print("Similarities computed in {}".format(dt.now() - tic))
    return similarities


columnstodrop = ["title", "career_level", "discipline_id", "industry_id", "country", "region", "latitude", "longitude",
                 "employment", "created_at", "active_during_test"]
# # Computing the COO matrix
tagsparsematrix = createcoomatrix(items.drop(columnstodrop, 1).tags.values)
# # Computing similarities matrix
similarities_matrix = compute_similarities(tagsparsematrix)
# # Saving Matrix for future use
save_sparse_csc("similarities", similarities_matrix.tocsc())

# for user in samplesIds:
#     ratings = getuserratings(user)
#     for rating in ratings:
#         # item_sim_matrix = compute_item_similarities(items.drop(columnstodrop, 1)[items.id == rating].index.values[0], tagsparsematrix).tocsc()
#         print(item_sim_matrix.shape)
#         break
#     break

