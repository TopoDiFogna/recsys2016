import pandas as pd
import operator
from scipy.sparse import csc_matrix

from userprofile import createdictionary, getuserratings
from nointeractionscomputation import *
from tfidf import *

# Loading Data
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
samples = pd.read_csv("data/sample_submission.csv", header=0)
users = pd.read_table("data/user_profile.csv", sep="\t", header=0)
tagdf = pd.read_csv("tag_matrix.csv", header=0)
titledf = pd.read_csv("title_matrix.csv", header=0)

# End loading data

# Prepocessing data
user_ids = samples.user_id.values
items.fillna(value="0", inplace=True)
users.fillna(value="0", inplace=True)
available_items = items[items.active_during_test == 1].drop(
    ["active_during_test", "created_at", "latitude", "longitude"], axis=1)


# End of prepocessing data

def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csc(filename):
    loader = np.load(filename)
    return csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def get_tags_intersection(row, in_tags):
    if list(set(in_tags) & set(row.split(','))):
        count = len(list(set(in_tags) & set(row.split(','))))
        return count
    else:
        return 0


def get_comparison(value, comparison):
    if (value == comparison):
        return 1
    else:
        return 0


def get_jobroles(row):
    return row.jobroles.values.tolist()


def recommend_no_ratings(jobroles, rnr_available_items):
    # Dict containing {item_index: count}
    title_series = rnr_available_items['tags'].apply((lambda x: get_tags_intersection(x, jobroles)))
    tag_series = rnr_available_items['title'].apply((lambda x: get_tags_intersection(x, jobroles)))

    total_series = title_series + tag_series
    final_dict = total_series.to_dict()
    # Sort by count
    rnr_sorted_id = sorted(final_dict.items(), key=operator.itemgetter(1), reverse=True)
    # Save the first 5 elements
    recommendations = []
    for rnr_elem in rnr_sorted_id[:5]:
        recommendations.append(getitemsid(rnr_elem[0], rnr_available_items))
    return recommendations


def compute_comparison(value, dictionary, base):
    if value in dictionary:
        if base == 0:
            return dictionary[value]
        else:
            if dictionary[value] == 0:
                return 0
            else:
                return base ** dictionary[value]
    else:
        return 0


def compute_comparison_string(value, dictionary, base):
    if isinstance(value, str):
        splitted_string = value.split(",")
        summation = 0
        for string in splitted_string:
            summation += compute_comparison(string, dictionary, base)
        return summation
    else:
        return 0


def computescore(itemdf, titlesdict, tagsdict, attribdict, alreadyclickeditems):
    items_ids = itemdf["id"]
    itemdf = itemdf.drop("id", axis=1)
    columns_names = itemdf.columns
    for colunm in columns_names:
        if colunm == "tags":
            itemdf[colunm] = itemdf[colunm].map(lambda x: compute_comparison_string(x, tagsdict, 0))
        elif colunm == "title":
            itemdf[colunm] = itemdf[colunm].map(lambda x: compute_comparison_string(x, titlesdict, 0))
        else:
            element_dict = attribdict[colunm]
            itemdf[colunm] = itemdf[colunm].map(lambda x: compute_comparison(x, element_dict, 0), na_action=None)

    sum_series = itemdf["tags"] + itemdf["title"]
    dictionary = dict(zip(items_ids.values, sum_series.values))
    for item in alreadyclickeditems:
        if item in dictionary:
            dictionary[item] = 0
    return dictionary


def getitemsid(item_indexes, dataset):
    return dataset.loc[item_indexes].id


# questo metodo server per riordinare le recommendations con lo stesso ordine
def order_ratings(sorteddict, tagsdict, titlesdict, attribdict, availableitems):
    orderedratings = []
    while len(orderedratings) < 5:
        maxvalue = sorteddict[0][1]
        equalids = []
        for elem in sorteddict:
            if elem[1] == maxvalue:
                equalids.append(elem[0])
            if elem[1] < maxvalue:
                break
        sorteddict = sorteddict[len(equalids):]
        if len(equalids) > 1:
            item_selected = availableitems[availableitems.id.isin(equalids)]
            ids = item_selected["id"]
            item_selected = item_selected.drop("id", axis=1)
            # potresti avere dei problemi con questa base perchè uso una valutazione di tipo esponenziale
            # ho aggiunto questo controllo per evitare l'overflow di float
            base = 10

            for colunm in item_selected.columns:
                if colunm == "tags":
                    try:
                        item_selected[colunm] = item_selected[colunm].map(
                            lambda x: compute_comparison_string(x, tagsdict, base))
                    except:
                        item_selected[colunm] = 1.7976931348623157e+308
                elif colunm == "title":
                    try:
                        item_selected[colunm] = item_selected[colunm].map(
                            lambda x: compute_comparison_string(x, titlesdict, base))
                    except:
                        item_selected[colunm] = 1.7976931348623157e+308

            sum_series = item_selected["tags"] + item_selected["title"]
            sum_series = sum_series.sort_values(ascending=False)
            sum_indexes = sum_series.index
            for index in sum_indexes:
                orderedratings.append(ids[index])
        else:
            orderedratings.append(equalids[0])
    return orderedratings[:5]


# Main code of the script
total_tic = dt.now()
top_pop = [1053452, 2778525, 1244196, 1386412, 657183]
interaction_user_df = getinteractionusers(users, interactions)
tags = pd.Series(index=tagdf.id, data=np.arange(tagdf.index.size))
titles = pd.Series(index=titledf.id, data=np.arange(titledf.index.size))

# title_matrix, tag_matrix = createcoomatrix(items,tags,titles)
# tag_matrix = tag_matrix.tocsc()
# title_matrix = title_matrix.tocsc()
# print(tag_matrix.tocsr())
# print(title_matrix.tocsr())
# save_sparse_csc("titleMatrix",title_matrix)
# save_sparse_csc("tagMatrix",tag_matrix)
tag_matrix = load_sparse_csc("tagMatrix.npz")
title_matrix = load_sparse_csc("titleMatrix.npz")

with open("test.csv", "w") as f:
    f.write("user_id,recommended_items\n")
    for user in user_ids:
        tic = dt.now()
        titles_dict, tags_dict, attrib = createdictionary(user, interactions, items, tag_matrix, title_matrix, tags,
                                                          titles)
        alreadyClickedItems = getuserratings(user, interactions)
        recommended_ids = []
        if len(attrib) > 0:
            items_score = computescore(available_items, titles_dict, tags_dict, attrib, alreadyClickedItems)
            # Sort by score
            sorted_id = sorted(items_score.items(), key=operator.itemgetter(1), reverse=True)
            recommended_ids = order_ratings(sorted_id, tags_dict, titles_dict, attrib, available_items)
            print(recommended_ids)
        else:
            print("USER {} has no ratings, recommendations done based on jobroles".format(user))
            user_row = users[users.user_id == user]
            u_jobroles = get_jobroles(user_row)
            available_items_filtered = available_items
            if not (user_row["industry_id"].values[0] == 0):
                available_items_filtered = available_items_filtered[
                    available_items_filtered["industry_id"] == user_row["industry_id"].values[0]]
            elif not (user_row["discipline_id"].values[0] == 0):
                available_items_filtered = available_items_filtered[
                    available_items_filtered["discipline_id"] == user_row["discipline_id"].values[0]]
            recommended_ids = recommend_no_ratings(u_jobroles, available_items_filtered)
            print("\tjobroles: {}".format(u_jobroles))
            print("\trecommandations: {}".format(recommended_ids))
            i = 0
            while len(recommended_ids) < 5:
                recommended_ids.append(top_pop[i])
                i += 1
        f.write("{},{}\n".format(user, ' '.join(str(e) for e in recommended_ids)))
        print("User {} computed in {}".format(user, dt.now() - tic))

print("Process ended after {}".format(dt.now() - total_tic))
