import numpy as np
import pandas as pd
from collections import Counter
import operator
from datetime import datetime as dt

# # Loading Data
# interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
# samples = pd.read_csv("data/sample_submission.csv", header=0)
# items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
# users = pd.read_table("data/user_profile.csv", sep="\t", header=0)
# # End loading data
#
# # Prepocessing data
# user_ids = samples.user_id.values
# items.fillna(value="0", inplace=True)
# # End of prepocessing data


# Gets the ratings a user has performed dropping the duplicates and keeping the highest
def getuserratings(userid):
    sampleinteractions = interactions[interactions['user_id'] == userid].reset_index().drop("index", 1).drop(
        "created_at", 1).drop("user_id", 1)
    sampleinteractions = sampleinteractions.groupby(by='item_id', as_index=False).apply(
        lambda x: x.ix[x.interaction_type.idxmax()])
    if sampleinteractions.empty:
        return np.array([])
    else:
        return sampleinteractions.item_id.values


# Returns the tags from a list of items ordered by count values
def get_tags_ordered(items_rated):
    tags = np.array([])
    for tag in items_rated.tags.values:
        tags = np.append(tags, tag.split(","))
    most_common = most_common_attribute(tags)
    return most_common


# Orders the array by counting
def most_common_attribute(array):
    count = Counter(array)
    return [tag[0] for tag in count.most_common()]


def get_tags_intersection(row, tags):
    if list(set(tags) & set(row.split(','))):
        count = len(list(set(tags) & set(row.split(','))))
        return count
    else:
        return 0  # no intersection


# Given the index returns the item id in the dataset
def getitemsid(item_indexes, dataset):
    return dataset.loc[item_indexes].id


def recommend(rated_user_items, rated_tags):
    available_items = items[items.active_during_test == 1]
    for item_id in rated_user_items.id:
        available_items = available_items[available_items.id != item_id]
    # Dict containing {item_index: count}
    tags_dict = available_items['tags'].apply((lambda x: get_tags_intersection(x, rated_tags))).to_dict()
    # Sort by count
    sorted_id = sorted(tags_dict.items(), key=operator.itemgetter(1), reverse=True)
    # Save the first 5 elements
    recommendations = []
    for elem in sorted_id[:5]:
        recommendations.append(getitemsid(elem[0], available_items))
    return recommendations


def get_jobroles(row):
    return row.jobroles.values.tolist()


def recommend_no_ratings(jobroles):
    available_items = items[items.active_during_test == 1]
    # Dict containing {item_index: count}
    title_dict = available_items['tags'].apply((lambda x: get_tags_intersection(x, jobroles))).to_dict()
    # Sort by count
    sorted_id = sorted(title_dict.items(), key=operator.itemgetter(1), reverse=True)
    # Save the first 5 elements
    recommendations = []
    for elem in sorted_id[:5]:
        recommendations.append(getitemsid(elem[0], available_items))
    return recommendations

# Main code of the script
# total_tic = dt.now()
# top_pop = [1053452, 2778525, 1244196, 1386412, 657183]
# with open("test.csv", "w") as f:
#     f.write("user_id,recommended_items\n")
#     for user in user_ids:
#         tic = dt.now()
#         ratings = getuserratings(user)
#         rated_items = pd.DataFrame(columns=items.columns).astype(np.int32)
#         for rating in ratings:
#             if not ratings.size == 0:  # User has reted something
#                 rated_items = rated_items.append(items[items.id == rating], ignore_index=True)
#         if not rated_items.empty:  # User has some ratngs
#             rated_items_tags = get_tags_ordered(rated_items)
#             print("USER: {}".format(user))
#             print("\trated items: {}".format(rated_items.id.values.tolist()))
#             print("\ttags: {}".format(rated_items_tags))
#             recommended_ids = recommend(rated_items, rated_items_tags)
#             print("\trecommandations: {}".format(recommended_ids))
#             i = 0
#             while len(recommended_ids) < 5:
#                 recommended_ids.append(top_pop[i])
#                 i += 1
#         else:  # User has no ratings
#             print("USER {} has no ratings, recommendations done based on jobroles".format(user))
#             user_row = users[users.user_id == user]
#             u_jobroles = get_jobroles(user_row)
#             recommended_ids = recommend_no_ratings(u_jobroles)
#             print("\tjobroles: {}".format(u_jobroles))
#             print("\trecommandations: {}".format(recommended_ids))
#             i = 0
#             while len(recommended_ids) < 5:
#                 recommended_ids.append(top_pop[i])
#                 i += 1
#         f.write("{},{}\n".format(user, ' '.join(str(e) for e in recommended_ids)))
#         print("User {} computed in {}".format(user, dt.now() - tic))
# print("Process ended after {}".format(dt.now()-total_tic))
