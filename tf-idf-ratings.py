import pandas as pd
import numpy as np
import operator
from datetime import datetime as dt

from utils.userprofile import createdictionary, getuserratings, createdictionary_noratings
from utils.dataloading import load_sparse_csc
from utils.cfutils import get_top_n_similar_users


# Elevates the score of a item by a certain number
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


# Splits a string separated by commas and calculates the score for every new substring
def compute_comparison_string(value, dictionary, base):
    if isinstance(value, str):
        splitted_string = value.split(",")
        summation = 0
        for string in splitted_string:
            summation += compute_comparison(string, dictionary, base)
        return summation
    else:
        return 0


def computescore(itemdf, titlesdict, tagsdict, alreadyclickeditems, sorted_similar_items_dict):
    items_ids = itemdf["id"]
    itemdf = itemdf.drop("id", axis=1)
    columns_names = itemdf.columns
    for column in columns_names:
        if column == "tags":
            itemdf[column] = itemdf[column].map(lambda x: compute_comparison_string(x, tagsdict, 0))
        elif column == "title":
            itemdf[column] = itemdf[column].map(lambda x: compute_comparison_string(x, titlesdict, 0))

    sum_series = itemdf["tags"] + itemdf["title"]
    dictionary = dict(zip(items_ids.values, sum_series.values))
    for item in alreadyclickeditems:
        if item in dictionary:
            dictionary[item] = 0
    for item in dictionary:
        for item2 in sorted_similar_items_dict:
            if item == item2[0]:
                dictionary[item] *= item2[1]
    return dictionary


def computescore_noratings(itemdf, jobrolesdict):
    items_ids = itemdf["id"]
    itemdf = itemdf.drop("id", axis=1)
    columns_names = itemdf.columns
    for column in columns_names:
        if column == "tags":
            itemdf[column] = itemdf[column].map(lambda x: compute_comparison_string(x, jobrolesdict, 0))
        elif column == "title":
            itemdf[column] = itemdf[column].map(lambda x: compute_comparison_string(x, jobrolesdict, 0))

    sum_series = itemdf["tags"] + itemdf["title"]
    dictionary = dict(zip(items_ids.values, sum_series.values))
    return dictionary


# Reorders the items with the same score in a proper way so we can have better recommendations
def order_ratings(sorteddict, tagsdict, titlesdict, availableitems):
    orderedratings = []
    while len(orderedratings) < 5:  # While we have less than 5 items to recommend
        maxvalue = sorteddict[0][1]
        equalids = []
        for elem in sorteddict:
            if elem[1] == maxvalue:
                equalids.append(elem[0])  # Take all the item with the same score
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

            for colunm in item_selected.columns:  # Score again these items to better recommend them
                if colunm == "tags":
                    try:
                        item_selected[colunm] = item_selected[colunm].map(
                            lambda x: compute_comparison_string(x, tagsdict, base))
                    except ArithmeticError:
                        item_selected[colunm] = 1.7976931348623157e+308
                elif colunm == "title":
                    try:
                        item_selected[colunm] = item_selected[colunm].map(
                            lambda x: compute_comparison_string(x, titlesdict, base))
                    except ArithmeticError:
                        item_selected[colunm] = 1.7976931348623157e+308

            sum_series = item_selected["tags"] + item_selected["title"]
            sum_series = sum_series.sort_values(ascending=False)
            sum_indexes = sum_series.index
            for index in sum_indexes:
                orderedratings.append(ids[index])
        else:
            orderedratings.append(equalids[0])
    return orderedratings[:5]  # Return the 5 items with top scores


def order_ratings_nointeractions(sorteddict, jobrolesdict, availableitems):
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
            base = 10

            for colunm in item_selected.columns:
                if colunm == "tags":
                    try:
                        item_selected[colunm] = item_selected[colunm].map(
                            lambda x: compute_comparison_string(x, jobrolesdict, base))
                    except ArithmeticError:
                        item_selected[colunm] = 1.7976931348623157e+308
                elif colunm == "title":
                    try:
                        item_selected[colunm] = item_selected[colunm].map(
                            lambda x: compute_comparison_string(x, jobrolesdict, base))
                    except ArithmeticError:
                        item_selected[colunm] = 1.7976931348623157e+308

            sum_series = item_selected["tags"] + item_selected["title"]
            sum_series = sum_series.sort_values(ascending=False)
            sum_indexes = sum_series.index
            for index in sum_indexes:
                orderedratings.append(ids[index])
        else:
            orderedratings.append(equalids[0])
    # print(orderedratings)  # TODO remove this and filter more
    return orderedratings[:5]


# Gets every item a given user has already clicked and for every other similar user adds 2
# to this similar user score for every equal item he has clicked that is not already found, 1 otherwise
def create_similar_dict(userid, already_clicked, interactiondf, n):
    dictionary = {}
    top_n = get_top_n_similar_users(userid, n)  # Get similar users
    for similar_user in top_n:
        for item in list(set(getuserratings(similar_user, interactiondf))):
            if item not in already_clicked:
                if item not in dictionary:
                    dictionary[item] = 2  # This item has been clicked by both users but it has been seen
                                          # for the first time
                else:
                    dictionary[item] += 1   # This item has been clicked by both users but it has been already seen
    return dictionary


# Justs saves the number each item is recommended to copute the most recommended items
def save_top_recommended_items(recommendations):
    for item in recommendations:
        if item not in top_recommended_items:
            top_recommended_items[item] = 1
        else:
            top_recommended_items[item] += 1

total_tic = dt.now()
print("Loading data...")
# Loading Data
#
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
samples = pd.read_csv("data/sample_submission.csv", header=0)
users = pd.read_table("data/user_profile.csv", sep="\t", header=0)
tagdf = pd.read_csv("precomputedData/tag_matrix.csv", header=0)
titledf = pd.read_csv("precomputedData/title_matrix.csv", header=0)
jobrolesdf = pd.read_csv("precomputedData/jobrole_matrix.csv", header=0)
tag_matrix = load_sparse_csc("precomputedData/tagMatrix.npz")
title_matrix = load_sparse_csc("precomputedData/titleMatrix.npz")
jobroles_matrix = load_sparse_csc("precomputedData/jobrolesMatrix.npz")
#
# End loading data

loading_time = dt.now()
print("Data loaded in {}".format(loading_time - total_tic))
print("Prepocessing Data...")

# Prepocessing data
#
user_ids = samples.user_id.values
# Replaces null values with zeroes for easier computation
items.fillna(value="0", inplace=True)
users.fillna(value="0", inplace=True)
available_items = items[items.active_during_test == 1].drop(
    ["active_during_test", "created_at", "latitude", "longitude"], axis=1)  # Filtering out available items
tags = pd.Series(index=tagdf.id, data=np.arange(tagdf.index.size))
titles = pd.Series(index=titledf.id, data=np.arange(titledf.index.size))
jobroles = pd.Series(index=jobrolesdf.id, data=np.arange(jobrolesdf.index.size))
#
# End of prepocessing data

print("Ended preprocessing in {}".format(dt.now()-loading_time))
print("Starting recommending!\n")
top_recommended_items = {}

with open("test.csv", "w") as f:
    f.write("user_id,recommended_items\n")
    no_jobrole_user = []  # Here will be stored users that do not have interactions and do not have a jobrole specified
    for user in user_ids:
        tic = dt.now()
        print("User {}".format(user))
        # Creates a user profile based on past interactions
        titles_dict, tags_dict = createdictionary(user, interactions, items, tag_matrix, title_matrix, tags, titles)
        # Filters out the items that are already clicked by the user so we do not recommend them again
        alreadyClickedItems = getuserratings(user, interactions)
        recommended_ids = []  # List used to store the recommended items
        if len(titles_dict) > 0 or len(tags_dict) > 0:
            # Items clicked by similar users
            similar_dict = create_similar_dict(user, alreadyClickedItems, interactions, 21)
            # Sort similar items to that clicked by the user
            sorted_similar_items = sorted(similar_dict.items(), key=operator.itemgetter(1), reverse=True)

            # Items that can be interesting for the user
            # Sort by score
            items_score = computescore(available_items, titles_dict, tags_dict, alreadyClickedItems, sorted_similar_items)
            sorted_id = sorted(items_score.items(), key=operator.itemgetter(1), reverse=True)
            recommended_ids = order_ratings(sorted_id, tags_dict, titles_dict, available_items)
            print(recommended_ids)
            # Just saves the fact that we're recommending this items so we can then have the most recommended items
            save_top_recommended_items(recommended_ids)
            # Write to file
            f.write("{},{}\n".format(user, ' '.join(str(e) for e in recommended_ids)))
        else:
            print("USER {} has no ratings, recommendations done based on jobroles".format(user))
            user_jobroles = set(users[users["user_id"] == user].jobroles.values)  # Gets user jobroles
            if user_jobroles != {'0'}:  # If some jobroles are found
                print("Jobroles: {}".format(user_jobroles))
                jobroles_dict = createdictionary_noratings(user, users, jobroles_matrix, jobroles)  # Create a dictionary
                items_score = computescore_noratings(items, jobroles_dict)  # Give a score for every item in the dictionary
                # Sort by score
                sorted_id = sorted(items_score.items(), key=operator.itemgetter(1), reverse=True)
                # Reorder if the items have the same score
                recommended_ids = order_ratings_nointeractions(sorted_id, jobroles_dict, available_items)
                print(recommended_ids)
                # Just saves the fact that we're recommending this items so we can then have the most recommended items
                save_top_recommended_items(recommended_ids)
                f.write("{},{}\n".format(user, ' '.join(str(e) for e in recommended_ids)))
            else:  # If the user hasa no jobroles just delay the recommendations so we can give the most recommended items
                print("User {} has not jobroles, delaying recommendations".format(user))
                no_jobrole_user.append(user)
        print("User {} computed in {}\n".format(user, dt.now() - tic))

    print("\nCalculating top recommended items")
    sorted_recommended_items = sorted(top_recommended_items.items(), key=operator.itemgetter(1), reverse=True)
    top_5_recommended_items = sorted_recommended_items[:5]
    print("Top recommended items: " + ' '.join(str(e[0]) for e in top_5_recommended_items))
    print("Writing file for user with no jobroles")
    for user in no_jobrole_user:
        f.write("{},{}\n".format(user, ' '.join(str(e[0]) for e in top_5_recommended_items)))
print("Process ended after {}".format(dt.now() - total_tic))
