import pandas as pd
from datetime import datetime as dt
import numpy as np
from scipy.sparse import coo_matrix

interactions = pd.read_table("../data/interactions.csv", sep="\t", header=0)
items = pd.read_table("../data/item_profile.csv", sep="\t", header=0)
samples = pd.read_csv("../data/sample_submission.csv", header=0)
users = pd.read_table("../data/user_profile.csv", sep="\t", header=0)
tagdf = pd.read_csv("../precomputedData/tag_matrix.csv", header=0)
titledf = pd.read_csv("../precomputedData/title_matrix.csv", header=0)
jobrolesdf = pd.read_csv("../precomputedData/jobrole_matrix.csv", header=0)

rating_user_array = interactions.user_id.unique()

def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def create_user_rating_matrix() :
    items_series = pd.Series(index=items.id, data=np.arange(items.index.size))
    # users_array = usersdf.user_id.values
    print("Computing Sparse COO matrix")
    columns = []
    rows = []
    index = 0
    tic = dt.now()
    for user in rating_user_array:
        item_clicked = interactions[interactions["user_id"] == user].item_id.unique()
        for item in item_clicked:
            rows.append(index)
            columns.append(items_series.loc[int(item)])
        index += 1
    data = np.ones_like(columns)
    user_rating_matrix = coo_matrix((data, (rows, columns)), shape=(rating_user_array.size, items_series.size))
    print("Coo matrix computed in: {}".format(dt.now() - tic))

    user_rating_matrix = user_rating_matrix.tocsc()
    save_sparse_csc("../precomputedData/user_rating_matrix", user_rating_matrix)



