import pandas as pd
from datetime import datetime as dt
import numpy as np
from scipy.sparse import coo_matrix
from utils.dataloading import *
from scipy.sparse import vstack
import math as m

# items = pd.read_table("../data/item_profile.csv", sep="\t", header=0)
# samples = pd.read_csv("../data/sample_submission.csv", header=0)
# users = pd.read_table("../data/user_profile.csv", sep="\t", header=0)
# tagdf = pd.read_csv("../precomputedData/tag_matrix.csv", header=0)
# titledf = pd.read_csv("../precomputedData/title_matrix.csv", header=0)
# jobrolesdf = pd.read_csv("../precomputedData/jobrole_matrix.csv", header=0)
interactions = pd.read_table("../data/interactions.csv", sep="\t", header=0)

matrix_similarity = load_sparse_csc("../precomputedData/userRatingSimilarity.npz")
rating_user_array = interactions.user_id.unique().tolist()

def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def create_user_rating_matrix() :
    items_series = pd.Series(index=items.id, data=np.arange(items.index.size))
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

def normalize_user_rating_matrix():
    user_rating_matrix = load_sparse_csc("../precomputedData/user_rating_matrix.npz").tocsr()
    lenght_vector = np.squeeze(np.asarray(user_rating_matrix.sum(axis=1)))
    index = 0
    user_rating_matrix_Normalized = user_rating_matrix.getrow(index) / m.sqrt(lenght_vector.item(index))
    lenght_vector = np.delete(lenght_vector,index)
    for item in lenght_vector :
        index = index + 1
        matrix_row = user_rating_matrix.getrow(index)
        if(item == 0) :
            user_rating_matrix_Normalized = vstack([user_rating_matrix_Normalized, matrix_row])
        else:
            user_rating_matrix_Normalized = vstack([user_rating_matrix_Normalized, matrix_row/m.sqrt(item)])

    save_sparse_csc("../precomputedData/userRatingsMatrixNormalized.npz",user_rating_matrix_Normalized.tocsc())

def create_user_rating_matrix_similarity():
    user_rating_matrix = load_sparse_csc("../precomputedData/userRatingsMatrixNormalized.npz")
    num_rows = user_rating_matrix.shape[0]
    rows_array = np.arange(num_rows)

    matrix_similarity = np.dot(user_rating_matrix.getrow(0),user_rating_matrix.T)
    for index in rows_array :
        tic = dt.now()
        if (index != 0) :
            new_row = np.dot(user_rating_matrix.getrow(index),user_rating_matrix.T)
            matrix_similarity = vstack([matrix_similarity, new_row])
        print("user {} computed in: {}".format(index,dt.now() - tic))

    save_sparse_csc("../precomputedData/userRatingSimilarity.npz",matrix_similarity.tocsc())

def get_top_n_similar_users(user_id, n):
    top_indexes = []
    user_index = rating_user_array.index(user_id)
    user_row = np.squeeze(matrix_similarity.getrow(user_index).toarray())
    np.put(user_row, user_index, 0)
    top_indexes = user_row.argsort()[-n:][::-1]
    result = []
    for index in top_indexes:
        target = rating_user_array[index]
        result.append(target)
    return result
