import pandas as pd
from datetime import datetime as dt
import numpy as np
from scipy.sparse import coo_matrix
from utils.dataloading import load_sparse_csc
from scipy.sparse import vstack
import math as m

# items = pd.read_table("../data/item_profile.csv", sep="\t", header=0)
# samples = pd.read_csv("../data/sample_submission.csv", header=0)
# users = pd.read_table("../data/user_profile.csv", sep="\t", header=0)
# tagdf = pd.read_csv("../precomputedData/tag_matrix.csv", header=0)
# titledf = pd.read_csv("../precomputedData/title_matrix.csv", header=0)
# jobrolesdf = pd.read_csv("../precomputedData/jobrole_matrix.csv", header=0)
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)

# Precomputed data for the user similarity
matrix_similarity = load_sparse_csc("precomputedData/userRatingSimilarity_IP.npz").tocsr()
rating_user_array = interactions.user_id.unique().tolist()


def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


# Creates a coo matrix for the user rating, used after to create a similarity matrix
def create_user_rating_matrix():
    items_series = pd.Series(index=items.id, data=np.arange(items.index.size))
    print("Computing Sparse COO matrix")
    columns = []
    rows = []
    data = []
    index = 0
    tic = dt.now()
    for user in rating_user_array:
        interaction_filtered = interactions[interactions["user_id"] == user]  # Gets the user interaction
        item_clicked = interaction_filtered.item_id.unique()
        for item in item_clicked:
            rows.append(index)
            columns.append(items_series.loc[int(item)])
            # Saves every user interaction with it's value
            data.append(interaction_filtered[interaction_filtered["item_id"] == item].interaction_type.values.size)
        index += 1
        print("User computed in {}".format(dt.now() - tic))
    user_rating_matrix = coo_matrix((data, (rows, columns)), shape=(len(rating_user_array), items_series.size))
    print("Coo matrix computed in: {}".format(dt.now() - tic))

    user_rating_matrix = user_rating_matrix.tocsc()
    save_sparse_csc("../precomputedData/user_rating_matrix_IP", user_rating_matrix)


# Normalization of the user rating matrix for meaningful results
def normalize_user_rating_matrix():
    user_rating_matrix = load_sparse_csc("../precomputedData/user_rating_matrix.npz").tocsr()
    lenght_vector = np.squeeze(np.asarray(user_rating_matrix.sum(axis=1)))
    index = 0
    user_rating_matrix_Normalized = user_rating_matrix.getrow(index) / m.sqrt(lenght_vector.item(index))
    lenght_vector = np.delete(lenght_vector, index)
    for item in lenght_vector:
        index = index + 1
        matrix_row = user_rating_matrix.getrow(index)
        # We do the computation line per line to save memory at a cost of time
        if (item == 0):
            user_rating_matrix_Normalized = vstack([user_rating_matrix_Normalized, matrix_row])
        else:
            user_rating_matrix_Normalized = vstack([user_rating_matrix_Normalized, matrix_row / m.sqrt(item)])

    save_sparse_csc("../precomputedData/userRatingsMatrixNormalized.npz", user_rating_matrix_Normalized.tocsc())


# Creates a user rating matrix based on previous saved matrix using the dot product (Inner Product method)
def create_user_rating_matrix_similarity():
    user_rating_matrix = load_sparse_csc("../precomputedData/user_rating_matrix_IP.npz")
    num_rows = user_rating_matrix.shape[0]
    rows_array = np.arange(num_rows)

    matrix_similarity = np.dot(user_rating_matrix.getrow(0), user_rating_matrix.T)  # Here is computed the dot product
    for index in rows_array:
        tic = dt.now()
        if (index != 0):
            new_row = np.dot(user_rating_matrix.getrow(index), user_rating_matrix.T)
            matrix_similarity = vstack([matrix_similarity, new_row])  # We do the computation line per line to save memory
                                                                        # at a cost of time
        print("user {} computed in: {}".format(index, dt.now() - tic))

    save_sparse_csc("../precomputedData/userRatingSimilarity_IP.npz", matrix_similarity.tocsc())


# Gets the top n similar users from the previously created similarity matrix
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
