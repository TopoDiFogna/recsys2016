import numpy as np
from math import sqrt, log


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
