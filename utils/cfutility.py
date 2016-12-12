import pandas as pd
#import math as m
from utils.dataloading import *
import numpy as np
#from scipy.sparse import vstack

def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


users = pd.read_table('../data/user_profile.csv', header=0, sep="\t")
interactions = pd.read_table('../data/interactions.csv', header=0, sep="\t")
samples = pd.read_csv("../data/sample_submission.csv", header=0)
rating_user_id = interactions.user_id.unique()
samples_id = samples.user_id

users_no_ratings = users[~ users["user_id"].isin(rating_user_id)]
users_no_ratings = users_no_ratings[users_no_ratings["user_id"].isin(samples_id)]
users_ratigns = users[users["user_id"].isin(rating_user_id)]

# job_roles_matrix = load_sparse_csc("../precomputedData/jobrolesMatrix.npz")

# indexes_no_ratings = users_no_ratings.index.tolist()
# indexes_ratings = users_ratigns.index.tolist()

# matrix_no_ratings_job_roles = job_roles_matrix.getrow(indexes_no_ratings[0])
# indexes_no_ratings.pop(0)
# for index in indexes_no_ratings :
#     matrix_no_ratings_job_roles = vstack([matrix_no_ratings_job_roles,job_roles_matrix.getrow(index)])
#
# matrix_no_ratings_job_roles = matrix_no_ratings_job_roles.tocsc()
# save_sparse_csc("../precomputedData/jobrolesNoRatingsMatrix.npz",matrix_no_ratings_job_roles)
#
# matrix_ratings_job_roles = job_roles_matrix.getrow(indexes_ratings[0])
# indexes_ratings.pop(0)
# for index in indexes_ratings :
#     matrix_ratings_job_roles = vstack([matrix_ratings_job_roles,job_roles_matrix.getrow(index)])
#
# matrix_ratings_job_roles = matrix_ratings_job_roles.tocsc()
# save_sparse_csc("../precomputedData/jobrolesRatingsMatrix.npz",matrix_ratings_job_roles)
#
# matrix_no_ratings_job_roles = load_sparse_csc("../precomputedData/jobrolesNoRatingsMatrix.npz")
# matrix_ratings_job_roles = load_sparse_csc("../precomputedData/jobrolesRatingsMatrix.npz")
# print(matrix_no_ratings_job_roles.shape)
# print(matrix_ratings_job_roles.shape)
#
# matrix_ratings_job_roles = matrix_ratings_job_roles.tocsr()
# lenght_vector = np.squeeze(np.asarray(matrix_ratings_job_roles.sum(axis=1)))
# index = 0
# matrix_ratings_job_roles_Normalized = matrix_ratings_job_roles.getrow(index) / m.sqrt(lenght_vector.item(index))
# lenght_vector = np.delete(lenght_vector,index)
# for item in lenght_vector :
#     index = index + 1
#     matrix_row = matrix_ratings_job_roles.getrow(index)
#     if(item == 0) :
#         matrix_ratings_job_roles_Normalized = vstack([matrix_ratings_job_roles_Normalized, matrix_row])
#     else:
#         matrix_ratings_job_roles_Normalized = vstack([matrix_ratings_job_roles_Normalized, matrix_row/m.sqrt(item)])
#
# save_sparse_csc("../precomputedData/jobrolesRatingsMatrixNormalized.npz",matrix_ratings_job_roles_Normalized.tocsc())
#
# matrix_no_ratings_job_roles = matrix_no_ratings_job_roles.tocsr()
# lenght_vector = np.squeeze(np.asarray(matrix_no_ratings_job_roles.sum(axis=1)))
# index = 0
# matrix_no_ratings_job_roles_Normalized = matrix_no_ratings_job_roles.getrow(index) / m.sqrt(lenght_vector.item(index))
# lenght_vector = np.delete(lenght_vector,index)
# for item in lenght_vector :
#     index = index + 1
#     matrix_row = matrix_no_ratings_job_roles.getrow(index)
#     if(item == 0) :
#         matrix_no_ratings_job_roles_Normalized = vstack([matrix_no_ratings_job_roles_Normalized, matrix_row])
#     else:
#         matrix_no_ratings_job_roles_Normalized = vstack([matrix_no_ratings_job_roles_Normalized, matrix_row/m.sqrt(item)])
#
# save_sparse_csc("../precomputedData/jobrolesNoRatingsMatrixNormalized.npz",matrix_no_ratings_job_roles_Normalized.tocsc())
#
# matrix_no_ratings_job_roles = load_sparse_csc("../precomputedData/jobrolesNoRatingsMatrixNormalized.npz")
# matrix_ratings_job_roles = load_sparse_csc("../precomputedData/jobrolesRatingsMatrixNormalized.npz")
#
# print(matrix_no_ratings_job_roles.shape)
# print(matrix_ratings_job_roles.shape)
#
# num_rows = matrix_no_ratings_job_roles.shape[0]
# rows_array = np.arange(num_rows)
#
# matrix_similarity = np.dot(matrix_no_ratings_job_roles.getrow(0),matrix_ratings_job_roles.T)
# for index in rows_array :
#     if (index != 0) :
#         new_row = np.dot(matrix_no_ratings_job_roles.getrow(index),matrix_ratings_job_roles.T)
#         matrix_similarity = vstack([matrix_similarity, new_row])
#
# save_sparse_csc("../precomputedData/jobrolesSimilarity.npz",matrix_similarity.tocsc())

matrix_similarity = load_sparse_csc("../precomputedData/jobrolesSimilarity.npz")
users_no_ratings = users_no_ratings.reset_index()

def compute_comparison(value, compareTo):
    if (compareTo == 0) :
        return 0
    if value == compareTo:
        return 1
    else:
        return 0

def get_top_n_similar_users(user_id, n) :
    user_row = users[users["user_id"] == user_id]
    user_jobroles = user_row.jobroles.values.item(0)
    splitted_string = user_jobroles.split(",")
    if (len(splitted_string) != 0):
        user_index = users_no_ratings[users_no_ratings["user_id"] == user_id].index.tolist()[0]
        user_row = matrix_similarity.getrow(user_index).toarray()
        user_row = -np.sort(-user_row)
        user_row = np.squeeze(user_row).tolist()
        return user_row[:n]
    else:
        user_to_find = users_ratigns.copy().reset_index(drop=True)
        userdf = user_to_find .drop(["user_id","jobroles","country","region","experience_n_entries_class","experience_years_experience","edu_degree","edu_fieldofstudies","experience_years_in_current"], axis=1)
        columns_names = userdf.columns
        for column in columns_names:
            userdf[column] = userdf[column].map(lambda x: compute_comparison(x, user_row[column].values.item(0)))
        sum_series = userdf["career_level"] + userdf["discipline_id"] + userdf["industry_id"]
        sum_series = sum_series.sort_values(ascending=False)
        top_indexes = sum_series.index.tolist()[:n]
        result =[]
        for index in top_indexes :
            target = user_to_find.iloc[[index]]
            result.append(target["user_id"].values.item(0))
        return result

# users_no_ratings_ids = users_no_ratings.user_id.values
# print(get_top_n_similar_users(users_no_ratings_ids.item(0),5))