import pandas as pd
import numpy as np
from datetime import datetime as dt


def square(x): return x ** 2

items = pd.read_table("normalized.csv", header=0, sep=",")
items.sort(columns="id",axis=0,inplace=True)
filteredItems = items[items.active_during_test == 1].reset_index()
sample = pd.read_table("data/sample_submission.csv", sep=",", header=0)
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)

# TODO prendere in inglesso gli utenti sample
colFuckingName = sample.columns[0]
sample = sample.rename(columns={colFuckingName: 'user_id'})
sampleIds = sample.user_id.values  # array contenente i sample user_id


# TODO cercare i lavori con iterazione prendendo quelli con rating massimo
def getuserratings(userid):
    sampleinteractions = interactions.loc[interactions['user_id'] == userid].reset_index().drop("index", 1).drop("created_at", 1)
    sampleinteractions = sampleinteractions.groupby(by='item_id', as_index=False).apply(lambda x: x.ix[x.interaction_type.idxmax()])
    sampleRatings=sampleinteractions.interaction_type.values
    print(sampleRatings)
    print(type(sampleRatings))
    sampleItems=items[items["id"].isin(sampleinteractions.item_id.values)].reset_index()
    inversesampleitems = filteredItems[~filteredItems['id'].isin(sampleinteractions.item_id.values)].reset_index()
    return [sampleItems, inversesampleitems]

tic = dt.now()
print(dt.now()-tic)

sampleItems, inverseSampleItems = getuserratings(2835411)


# tic = dt.now()
#
# for index, first_row in sampleItems.iterrows():  # solo quelli cliccati
#     for index, second_row in inverseSampleItems.iterrows():
#
#         s_tag = 0
#         s_title = 0
#         i_tag_len = 0
#         j_tag_len = 0
#         i_title_len = 0
#         j_title_len = 0
#         s_i_j = 0
#
#         if not (isinstance(first_row["tags"], float)):
#             a = np.array(first_row["tags"].split(","))
#             j_tag_len = len(a)
#             if not (isinstance(second_row["tags"], float)):
#                 b = np.array(second_row["tags"].split(","))
#                 i_tag_len = len(b)
#                 s_tag = len(np.intersect1d(a, b))
#             else:
#                 pass
#         else:
#             pass
#
#         if not (isinstance(first_row["title"], float)):
#             c = np.array(first_row["title"].split(","))
#             j_title_len = len(c)
#             if not (isinstance(second_row["title"], float)):
#                 d = np.array(second_row["title"].split(","))
#                 i_title_len = len(d)
#                 s_title = len(np.intersect1d(c, d))
#             else:
#                 pass
#         else:
#             pass
#         i_series = first_row.drop(["title", "tags"])
#         j_series = second_row.drop(["title", "tags"])
#
#         result_series = i_series.multiply(j_series)
#         result_series = result_series.drop(["index","id","created_at","active_during_test"])
#         result_series["similarity"] = result_series.sum()
#         print(result_series)
#         similarity = result_series["similarity"]
#
#         i_series = i_series.apply(square)
#         j_series = j_series.apply(square)
#         i_series["sum"] = i_series.sum()
#         j_series["sum"] = j_series.sum()
#         i_sum = i_series["sum"]
#         j_sum = j_series["sum"]
#         denominator = (i_sum + i_tag_len + i_title_len) ** 0.5 * (j_sum + j_title_len + j_tag_len) ** 0.5
#
#         similarity = similarity / denominator + s_title / denominator + s_tag / denominator
#         print(similarity)
#         print("Similarity computed in: {}".format(dt.now() - tic))
#     print("Total similarity computed in: {}".format(dt.now() - tic))
# TODO calcolare i rating

# TODO sputare i 5 più alti
