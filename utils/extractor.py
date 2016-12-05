import pandas as pd
from datetime import datetime as dt
import numpy as np
from scipy.sparse import coo_matrix


def createcoomatrix(usersdf, dataframe):
    jobroles_array = usersdf.jobroles.values
    users_array = usersdf.user_id.values
    print("Computing Sparse COO matrix")
    columns = []
    rows = []
    index = 0
    tic = dt.now()
    for jobrole_element in jobroles_array:
        if jobrole_element != "0":
            jobrole_element = jobrole_element.split(',')
            for jobrole in jobrole_element:
                rows.append(index)
                columns.append(dataframe.loc[int(jobrole)])
        index += 1
    data = np.ones_like(columns)
    jobrole_matrix = coo_matrix((data, (rows, columns)), shape=(users_array.size, dataframe.size))
    print("Coo matrix computed in: {}".format(dt.now() - tic))

    return jobrole_matrix


def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


users = pd.read_table('../data/user_profile.csv', header=0, sep="\t")
jobrolesSeries = pd.Series()
users.fillna('0', inplace=True)
value = 0
for jobroleRow in users.jobroles:
    for user_jobrole in jobroleRow.split(','):
        if (not (user_jobrole in jobrolesSeries.index)) and not (user_jobrole == '0'):
            jobrolesSeries.set_value(user_jobrole, value)
            value += 1

with open("../precomputedData/jobrole_matrix.csv", "w") as f:
    f.write("id,index\n")
    jobrolesSeries.to_csv(f, sep=',')

jobrolesdf = pd.read_csv("../precomputedData/jobrole_matrix.csv", header=0)
jobroles = pd.Series(index=jobrolesdf.id, data=np.arange(jobrolesdf.index.size))
jobroles_matrix = createcoomatrix(users, jobroles)
jobroles_matrix = jobroles_matrix.tocsc()
save_sparse_csc("../precomputedData/jobroleMatrix", jobroles_matrix)


items = pd.read_table("../data/item_profile.csv", header=0, sep="\t")
titleSeries = pd.Series()
items.fillna("0", inplace=True)
value = 0
for titleRow in items.title:
    for title in titleRow.split(","):
        if (not (title in titleSeries.index)) and not (title == "0"):
            titleSeries.set_value(title, value)
            value += 1

with open("../precomputedData/title_matrix.csv", "w") as f:
    f.write("id,index\n")
    titleSeries.to_csv(f, sep=',')

titledf = pd.read_csv("../precomputedData/title_matrix.csv", header=0)
titles = pd.Series(index=titledf.id, data=np.arange(titledf.index.size))
titles_matrix = createcoomatrix(users, titles)
titles_matrix = titles_matrix.tocsc()
save_sparse_csc("../precomputedData/titleMatrix", titles_matrix)


items = pd.read_table("../data/item_profile.csv", header=0, sep="\t")
tagSeries = pd.Series()
items.fillna("0", inplace=True)
value = 0
for tagRow in items.tags:
    for tag in tagRow.split(","):
        if (not (tag in tagSeries.index)) and not (tag == "0"):
            tagSeries.set_value(tag, value)
            value += 1

with open("../precomputedData/tag_matrix.csv", "w") as f:
    f.write("id,index\n")
    tagSeries.to_csv(f, sep=',')

tagdf = pd.read_csv("../precomputedData/tag_matrix.csv", header=0)
tags = pd.Series(index=titledf.id, data=np.arange(titledf.index.size))
tags_matrix = createcoomatrix(users, tags)
tags_matrix = tags_matrix.tocsc()
save_sparse_csc("../precomputedData/tagMatrix", tags_matrix)
