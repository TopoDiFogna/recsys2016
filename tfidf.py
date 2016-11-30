import numpy as np
from datetime import datetime as dt
from scipy.sparse import coo_matrix
from math import *


# questa funzione prende in ingresso una matrice csr NumItem x NumAttr e restituisce la matrice dei df
def tfcomputing(attribute_matrix, row_index, col_index):
    lenght_vector = np.squeeze(np.asarray(attribute_matrix.sum(axis=1)))
    num = attribute_matrix[row_index, col_index]
    dem = sqrt(lenght_vector[row_index])
    tf_result = num / dem
    return tf_result


def idfcomputing(attribute_matric, col_index):
    num_doc = attribute_matric.shape[0]
    idf_array = np.squeeze(np.asarray(attribute_matric.sum(axis=0)))
    idf_value = log(num_doc / idf_array[col_index], 10)
    return idf_value


def tf_idfcomputing(attribute_matrix, row_index, col_index):
    tf_value = tfcomputing(attribute_matrix, row_index, col_index)
    idf_value = idfcomputing(attribute_matrix, col_index)
    tf_idf_matrix = tf_value * idf_value
    return tf_idf_matrix


def createcoomatrix(itemsdf, tags, titles):
    columnstodrop_tag = ["title", "career_level", "discipline_id", "industry_id", "country", "region", "latitude",
                         "longitude",
                         "employment", "created_at", "active_during_test"]
    columnstodrop_title = ["tags", "career_level", "discipline_id", "industry_id", "country", "region", "latitude",
                           "longitude",
                           "employment", "created_at", "active_during_test"]
    items_tags_dropped = itemsdf.drop(columnstodrop_tag, axis=1, inplace=False)
    items_title_dropped = itemsdf.drop(columnstodrop_title, axis=1, inplace=False)
    tags_array = items_tags_dropped.tags.values
    title_array = items_title_dropped.title.values
    item_array = itemsdf.id.values
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
    print("Tags matrix computed in: {}".format(dt.now() - tic))
    tag_matrix = coo_matrix((data, (rows, columns)), shape=(item_array.size, tags.size))

    columns = []
    rows = []
    index = 0
    tic = dt.now()
    for tagelement in title_array:
        if tagelement != "0":
            tagelement = tagelement.split(',')
            for tag in tagelement:
                rows.append(index)
                columns.append(titles.loc[int(tag)])
        index += 1

    data = np.ones_like(columns)
    print("Title matrix computed in: {}".format(dt.now() - tic))
    title_matrix = coo_matrix((data, (rows, columns)), shape=(item_array.size, titles.size))

    return tag_matrix, title_matrix
