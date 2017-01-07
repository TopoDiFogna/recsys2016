import numpy as np
from math import sqrt, log
from utils.dataloading import load_sparse_csc
from scipy.sparse import vstack
def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

# questa funzione prende in ingresso una matrice csr NumItem x NumAttr e restituisce la matrice dei df
def tfcomputing(attribute_matrix, row_index, col_index):
    lenght_vector = np.squeeze(np.asarray(attribute_matrix.sum(axis=1)))
    num = attribute_matrix[row_index, col_index]
    dem = sqrt(lenght_vector[row_index])
    tf_result = num / dem
    return tf_result


def idfcomputing(attribute_matrix, col_index):
    num_doc = attribute_matrix.shape[0]
    idf_array = np.squeeze(np.asarray(attribute_matrix.sum(axis=0)))
    idf_value = log(num_doc / idf_array[col_index], 10)
    return idf_value


def tf_idfcomputing(attribute_matrix, row_index, col_index):
    tf_value = tfcomputing(attribute_matrix, row_index, col_index)
    idf_value = idfcomputing(attribute_matrix, col_index)
    tf_idf_matrix = tf_value * idf_value
    return tf_idf_matrix

# tag_matrix = load_sparse_csc("../precomputedData/tagMatrix.npz")
# title_matrix = load_sparse_csc("../precomputedData/titleMatrix.npz")
#
# lenght_vector = np.squeeze(np.asarray(title_matrix.sum(axis=1)))
# num_doc = title_matrix.shape[0]
# idf_array = np.squeeze(np.asarray(title_matrix.sum(axis=0)))
# value = idf_array/num_doc
# value = np.power(value,-1)
# idf_array_value = log(value, 10)
# dem = sqrt(lenght_vector)
#
# rows_array = np.arange(num_doc)
# first_row = title_matrix.getrow(0)
# tf_row = first_row/dem[0]
# matrix_tf_idf =
# for index in rows_array :
#     tic = dt.now()
#     if (index != 0) :
#         new_row = np.dot(user_rating_matrix.getrow(index),user_rating_matrix.T)
#         matrix_szmilarity = vstack([matrix_similarity, new_row])
#     print("user {} computed in: {}".format(index,dt.now() - tic))
#
#     save_sparse_csc("../precomputedData/userRatingSimilarity.npz",matrix_similarity.tocsc())
#
# tf_result = num / dem


