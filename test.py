import numpy as np
import pandas as pd
import re
from scipy.sparse import coo_matrix, hstack
from scipy import spatial
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


# Gets the ratings a user has performed dropping the duplicates and keeping the highest
def getuserratings(userid):
    sampleinteractions = interactions.loc[interactions['user_id'] == userid].reset_index().drop("index", 1).drop(
        "created_at", 1).drop("user_id", 1)
    sampleinteractions = sampleinteractions.groupby(by='item_id', as_index=False).apply(
        lambda x: x.ix[x.interaction_type.idxmax()])
    if sampleinteractions.empty:
        return np.array([])
    else:
        return sampleinteractions.interaction_type.values


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
    return coo_matrix((data, (rows, columns)), shape=(itemArray.size, tags.size))


columnstodrop = ["title", "career_level", "discipline_id", "industry_id", "country", "region", "latitude", "longitude",
                 "employment", "created_at", "active_during_test"]
# Computing the COO matrix
tagsparsematrix = createcoomatrix(items.drop(columnstodrop, 1).tags.values).tocsr()
print(tagsparsematrix[:10])

similarities_sparse = cosine_similarity(tagsparsematrix[:100000], dense_output=False)
print(similarities_sparse)


