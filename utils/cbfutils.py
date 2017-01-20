import pandas as pd
from datetime import datetime as dt
import numpy as np
from scipy.sparse import coo_matrix
from utils.dataloading import *
from scipy.sparse import vstack

import math as m

items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
samples = pd.read_csv("data/sample_submission.csv", header=0)
#users = pd.read_table("../data/user_profile.csv", sep="\t", header=0)
#tagdf = pd.read_csv("../precomputedData/tag_matrix.csv", header=0)
#titledf = pd.read_csv("../precomputedData/title_matrix.csv", header=0)
# jobrolesdf = pd.read_csv("../precomputedData/jobrole_matrix.csv", header=0)
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)

matrix_similarity = load_sparse_csc("precomputedData/userRatingCBF.npz").tocsr()
samples_user = samples.user_id.values
rating_user_array = interactions[interactions.user_id.isin(samples_user)].user_id.unique().tolist()
active_item_array = items[items["active_during_test"] == 1].id.values.tolist()

def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def create_attribute_series():
    attributeSeries = pd.Series()
    items.fillna("0", inplace=True)
    value = 0
    for tagRow in items.tags:
        for tag in tagRow.split(","):
            if (not (tag in attributeSeries.index)) and not (tag == "0"):
                attributeSeries.set_value(tag, value)
                value += 1
    for titleRow in items.title:
        for title in titleRow.split(","):
            if (not (title in attributeSeries.index)) and not (title == "0"):
                attributeSeries.set_value(title, value)
                value += 1

    with open("../precomputedData/attribute_series.csv", "w") as f:
        f.write("id,index\n")
        attributeSeries.to_csv(f, sep=',')

def create_item_content_matrix(itemsdf, attributeDataFrame):
    itemsdf.fillna("0", inplace=True)
    tags_array = itemsdf.tags.values
    titles_array = itemsdf.title.values
    items_array = itemsdf.id.values
    print("Computing Sparse COO matrix")
    columns = []
    rows = []
    index = 0
    tic = dt.now()
    length_vector = np.arange(0,items_array.size)

    for i in length_vector :
        title_element = titles_array[i]
        tag_element = tags_array[i]

        if title_element != "0":
            title_element = title_element.split(',')
            for title in title_element:
                rows.append(index)
                columns.append(attributeDataFrame.loc[int(title)])

        if tag_element != "0":
            tag_element = tag_element.split(',')
            for tag in tag_element:
                rows.append(index)
                columns.append(attributeDataFrame.loc[int(tag)])

        index += 1

    data = np.ones_like(columns)
    item_content_matrix = coo_matrix((data, (rows, columns)), shape=(items_array.size, attributeDataFrame.size))
    print("Coo matrix computed in: {}".format(dt.now() - tic))

    save_sparse_csc("../precomputedData/itemContentMatrix.npz",
                    item_content_matrix.tocsc())

def normalize_item_content_matrix():
    item_content_matrix = load_sparse_csc("../precomputedData/itemContentMatrix.npz").tocsr()
    lenght_vector = np.squeeze(np.asarray(item_content_matrix.sum(axis=1)))
    index = 0
    item_content_matrix_Normalized = item_content_matrix.getrow(index) / m.sqrt(lenght_vector.item(index))
    lenght_vector = np.delete(lenght_vector,index)
    for item in lenght_vector :
        index = index + 1
        matrix_row = item_content_matrix.getrow(index)
        if(item == 0) :
            item_content_matrix_Normalized = vstack([item_content_matrix_Normalized, matrix_row])
        else:
            item_content_matrix_Normalized = vstack([item_content_matrix_Normalized, matrix_row/m.sqrt(item)])

    save_sparse_csc("../precomputedData/itemContentMatrixNormalized.npz",item_content_matrix_Normalized.tocsc())

def create_item_content_matrix_similarity():
    tic = dt.now()
    item_content_matrix = load_sparse_csc("../precomputedData/itemContentMatrixNormalized.npz")
    submission_users = samples.user_id.values
    sub_interactions = interactions[interactions.user_id.isin(submission_users)]
    clicked_items = sub_interactions.item_id.unique()
    clicked_index = items[items.id.isin(clicked_items)].index.tolist()
    active_items_index = items[items["active_during_test"] == 1].index.tolist()
    clicked_items_matrix = item_content_matrix.tocsr()[clicked_index,:]
    active_items_matrix = item_content_matrix.tocsr()[active_items_index,:]
    active_items = items[items["active_during_test"] == 1].id.values

    active_series = pd.Series(active_items)
    clicked_series = pd.Series(clicked_items)

    matrix_similarity = np.dot(clicked_items_matrix,active_items_matrix.T)

    print(type(matrix_similarity))

    print("matrix_similarity computed in {}".format(dt.now() - tic))
    tic = dt.now()
    clicked_sub_items = sub_interactions[sub_interactions.item_id.isin(active_items)].item_id.unique()

    print(len(clicked_sub_items))

    col_array = []
    row_array = []

    for item in clicked_sub_items :
        active_index = active_series[active_series == item].index[0]
        clicked_ind = clicked_series[clicked_series == item].index[0]
        np.append(col_array,active_index)
        np.append(row_array,clicked_ind)

    print("zero array computed in {}".format(dt.now() - tic))

    tic= dt.now()

    matrix_similarity[row_array,col_array] = 0

    print("zeroMatrix computed in {}".format(dt.now() - tic))


    # matrix_similarity = np.dot(clicked_items_matrix.getrow(0),active_items_matrix.T)
    # for index in rows_array :
    #     tic = dt.now()
    #     if (index != 0) :
    #         new_row = np.dot(clicked_items_matrix.getrow(index),active_items_matrix.T)
    #         matrix_similarity = vstack([matrix_similarity, new_row])
    #     print("item {} computed in: {}".format(index,dt.now() - tic))

    return  matrix_similarity

def create_user_recc_matrix():
    user_rating_matrix = load_sparse_csc("../precomputedData/user_rating_matrix.npz")

    submission_users = samples.user_id.values
    user_interactions = interactions[interactions.user_id.isin(submission_users)].user_id.unique()
    user_index  = []

    for index in user_interactions :
        user_index.append(rating_user_array.index(index))

    sub_interactions = interactions[interactions.user_id.isin(submission_users)]
    clicked_items = sub_interactions.item_id.unique()
    clicked_index = items[items.id.isin(clicked_items)].index.tolist()

    user_index_matrix = user_rating_matrix.tocsr()[user_index, :]
    user_rating_matrix_final= user_index_matrix.tocsc()[:,clicked_index]

    num_rows = user_rating_matrix_final.shape[0]
    rows_array = np.arange(int(num_rows / 2))
    rows_array_button = np.arange(int(num_rows/2), num_rows)

    matrix_similarity = create_item_content_matrix_similarity()

    upperHalf = user_rating_matrix_final.tocsr()[rows_array,:]
    buttonHalf = user_rating_matrix_final.tocsr()[rows_array_button,:]

    upper_half_matrix = np.dot(upperHalf, matrix_similarity)
    button_half_matrix = np.dot(buttonHalf,matrix_similarity)
    matrix_cbf_recc = vstack([upper_half_matrix, button_half_matrix])

    #matrix_cbf_recc = np.dot(user_rating_matrix_final, matrix_similarity)
    # for index in rows_array:
    #     tic = dt.now()
    #     if (index != 0):
    #         new_row = np.dot(user_rating_matrix_final.getrow(index), matrix_similarity)
    #         matrix_cbf_recc = vstack([matrix_cbf_recc, new_row])
    #     print("user {} computed in: {}".format(index, dt.now() - tic))

    save_sparse_csc("../precomputedData/userRatingCBF.npz", matrix_cbf_recc.tocsc())

def get_top_n_similar_item(user_id, n):
    top_indexes = []
    user_index = rating_user_array.index(user_id)
    user_row = np.squeeze(matrix_similarity.getrow(user_index).toarray())
    top_indexes = user_row.argsort()[-n:][::-1]
    result = []
    for index in top_indexes:
        target = active_item_array[index]
        result.append(target)
    return result

