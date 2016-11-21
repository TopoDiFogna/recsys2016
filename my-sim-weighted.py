import pandas as pd
from datetime import datetime as dt
import operator
from userprofile import createdictionary, getuserratings
from nointeractionscomputation import *

# Loading Data
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
samples = pd.read_csv("data/sample_submission.csv", header=0)
users = pd.read_table("data/user_profile.csv", sep="\t", header=0)
# End loading data

# Prepocessing data
user_ids = samples.user_id.values
items.fillna(value="0", inplace=True)
users.fillna(value="0", inplace=True)
available_items = items[items.active_during_test == 1].drop(["active_during_test", "created_at"], axis=1)


# End of prepocessing data


def get_tags_intersection(row, in_tags):
    if list(set(in_tags) & set(row.split(','))):
        count = len(list(set(in_tags) & set(row.split(','))))
        return count
    else:
        return 0


def get_jobroles(row):
    return row.jobroles.values.tolist()


def recommend_no_ratings(jobroles, rnr_available_items):
    # Dict containing {item_index: count}
    title_dict = rnr_available_items['tags'].apply((lambda x: get_tags_intersection(x, jobroles))).to_dict()
    # Sort by count
    rnr_sorted_id = sorted(title_dict.items(), key=operator.itemgetter(1), reverse=True)
    # Save the first 5 elements
    recommendations = []
    for rnr_elem in rnr_sorted_id[:5]:
        recommendations.append(getitemsid(rnr_elem[0], rnr_available_items))
    return recommendations


def compute_comparison(value, dictionary):
    if value in dictionary:
        return dictionary[value]
    else:
        return 0


def compute_comparison_string(value, dictionary):
    if isinstance(value, str):
        splitted_string = value.split(",")
        summation = 0
        for string in splitted_string:
            summation += compute_comparison(string, dictionary)
        return summation
    else:
        return 0


def computescore(itemdf, titlesdict, tagsdict, attribdict, alreadyclickeditems):
    items_ids = itemdf["id"]
    itemdf = itemdf.drop("id", axis=1)
    columns_names = itemdf.columns
    for colunm in columns_names:
        if colunm == "tags":
            itemdf[colunm] = itemdf[colunm].map(lambda x: compute_comparison_string(x, tagsdict))
        elif colunm == "title":
            itemdf[colunm] = itemdf[colunm].map(lambda x: compute_comparison_string(x, titlesdict))
        else:
            element_dict = attribdict[colunm]
            itemdf[colunm] = itemdf[colunm].map(lambda x: compute_comparison(x, element_dict), na_action=None)
    sum_series = itemdf.sum(axis=1)
    dictionary = dict(zip(items_ids.values, sum_series.values))
    for item in alreadyclickeditems:
        if item in dictionary:
            dictionary[item] = 0
    return dictionary


def getitemsid(item_indexes, dataset):
    return dataset.loc[item_indexes].id


# Main code of the script
total_tic = dt.now()
top_pop = [1053452, 2778525, 1244196, 1386412, 657183]
interaction_user_df= getinteractionusers(users,interactions)
with open("test.csv", "w") as f:
    f.write("user_id,recommended_items\n")
    for user in user_ids:
        tic = dt.now()
        titles, tags, attrib = createdictionary(user, interactions, items)
        alreadyClickedItems = getuserratings(user, interactions)
        recommended_ids = []
        if len(attrib) > 0:
            # se questo è un dizionario in fprma {itemid: score} basta de-commentare le righe sotto ed è fatta
            items_score = computescore(available_items, titles, tags, attrib,
                                       alreadyClickedItems)
            # Sort by score
            sorted_id = sorted(items_score.items(), key=operator.itemgetter(1), reverse=True)
            # Save the first 5 elements
            for elem in sorted_id[:5]:
                recommended_ids.append(elem[0])
        else:
            print("USER {} has no ratings, recommendations done based on jobroles".format(user))
            user_row = users[users.user_id == user]
            # u_jobroles = get_jobroles(user_row)
            # recommended_ids = recommend_no_ratings(u_jobroles, available_items)
            # print("\tjobroles: {}".format(u_jobroles))
            # print("\trecommandations: {}".format(recommended_ids))
            # i = 0
            # while len(recommended_ids) < 5:
            #     recommended_ids.append(top_pop[i])
            #     i += 1
            attrdict, jobdict, edudict = create_dictionary_user(user_row)
            returndict = computenoratingssimilarity(interaction_user_df, jobdict, attrdict, edudict)
            sorted_id = sorted(returndict.items(), key=operator.itemgetter(1), reverse=True)
            simil_users = {}
            for elem in sorted_id[:10]:
                simil_users[elem[0]] = elem[1]
            recommended_ids=compute_recommendations(simil_users, interactions, available_items)
        f.write("{},{}\n".format(user, ' '.join(str(e) for e in recommended_ids)))
        print("User {} computed in {}".format(user, dt.now() - tic))

print("Process ended after {}".format(dt.now() - total_tic))
