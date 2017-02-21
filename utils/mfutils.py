import numpy as np
import pandas as pd
from utils.dataloading import load_sparse_csc
from implicit import alternating_least_squares
from scipy.sparse import coo_matrix
from datetime import datetime as dt

items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
# samples = pd.read_csv("../data/sample_submission.csv", header=0)
# users = pd.read_table("../data/user_profile.csv", sep="\t", header=0)
# tagdf = pd.read_csv("../precomputedData/tag_matrix.csv", header=0)
# titledf = pd.read_csv("../precomputedData/title_matrix.csv", header=0)
# jobrolesdf = pd.read_csv("../precomputedData/jobrole_matrix.csv", header=0)
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)

rating_user_array = interactions.user_id.unique().tolist()
user_factor = np.load("precomputedData/user_factor_matrix.npy")
item_factor = np.load("precomputedData/item_factor_matrix.npy")
non_active_items = items[items["active_during_test"] == 0].index.tolist()
item_list = items.id.values

def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
indptr=array.indptr, shape=array.shape)

def create_user_rating_matrix():
    items_series = pd.Series(index=items.id, data=np.arange(items.index.size))
    print("Computing Sparse COO matrix")
    columns = []
    rows = []
    data = []
    index = 0
    tic1 = dt.now()
    for user in rating_user_array:
        tic = dt.now()
        interaction_filtered = interactions[interactions["user_id"] == user]
        item_clicked = interaction_filtered.item_id.unique()
        for item in item_clicked:
            rows.append(index)
            columns.append(items_series.loc[int(item)])
            item_interaction =interaction_filtered[interaction_filtered["item_id"] == item]
            data.append(item_interaction.interaction_type.values.size)
        index += 1
        print("User computed in {}".format(dt.now() - tic))
    user_rating_matrix = coo_matrix((data, (rows, columns)), shape=(len(rating_user_array), items_series.size))
    print("Coo matrix computed in: {}".format(dt.now() - tic1))

    user_rating_matrix = user_rating_matrix.tocsc()
    save_sparse_csc("../precomputedData/user_rating_matrix_new", user_rating_matrix)

def bm25_weight(X, K1=100, B=0.8):
    """ Weighs each row of the sparse matrix of the data by BM25 weighting """
    # calculate idf per term (user)
    X = coo_matrix(X)
    N = X.shape[0]
    idf = np.log(float(N) / (1 + np.bincount(X.col)))

    # calculate length_norm per document (artist)
    row_sums = np.ravel(X.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X

def compute_factor_matrix(factors=1000, regularization=0.01,
                              iterations=50,
                              use_native=True,
                              dtype=np.float32,
                              cg=False):
    print("reading data from %s")
    start = dt.now()
    user_rating_matrix = load_sparse_csc("../precomputedData/user_rating_matrix_new.npz");
    print("read data file in %s", dt.now() - start)

    print("weighting matrix by bm25")
    weighted = bm25_weight(user_rating_matrix)

    print("calculating factors")
    start = dt.now()
    user_factors, item_factors = alternating_least_squares(weighted,
                                                             factors=factors,
                                                             regularization=regularization,
                                                             iterations=iterations,
                                                             use_native=use_native,
                                                             dtype=dtype,
                                                             use_cg=cg,
                                                             calculate_training_loss=True)

    print("calculated factors in %s", dt.now() - start)
    np.save("../precomputedData/user_factor_matrix",user_factors)
    np.save("../precomputedData/item_factor_matrix",item_factors)

def get_top_n_items(user_id, n,already_clicked_items):
    top_indexes = []
    user_index = rating_user_array.index(user_id)
    recommendations = np.dot(user_factor[user_index],item_factor.transpose())
    recommendations[non_active_items] = 0
    clicked_index = items[items.id.isin(already_clicked_items)].index.tolist()
    recommendations[clicked_index] = 0
    top_indexes = recommendations.argsort()[-n:][::-1]
    result = []
    for index in top_indexes:
        target = item_list[index]
        result.append(target)
    return result

# compute_factor_matrix()