import pandas as pd
import numpy as np
from datetime import datetime as dt
import operator
from userprofile import createdictionary


# Loading Data
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
samples = pd.read_csv("data/sample_submission.csv", header=0)
users = pd.read_table("data/user_profile.csv", sep="\t", header=0)
# End loading data

# Prepocessing data
user_ids = samples.user_id.values
items.fillna(value="0", inplace=True)
available_items = items[items.active_during_test == 1].drop(["active_during_test", "created_at"], axis=1)
# End of prepocessing data

def get_tags_intersection(row, tags):
    if list(set(tags) & set(row.split(','))):
        count = len(list(set(tags) & set(row.split(','))))
        return count
    else:
        return 0

def get_jobroles(row):
    return row.jobroles.values.tolist()

def recommend_no_ratings(jobroles,available_items):
    # Dict containing {item_index: count}
    title_dict = available_items['tags'].apply((lambda x: get_tags_intersection(x, jobroles))).to_dict()
    # Sort by count
    sorted_id = sorted(title_dict.items(), key=operator.itemgetter(1), reverse=True)
    # Save the first 5 elements
    recommendations = []
    for elem in sorted_id[:5]:
        recommendations.append(getitemsid(elem[0], available_items))
    return recommendations

def compute_comparison( value, dict) :
    if value in dict :
        return dict[value]
    else :
        return 0

def compute_comparison_string(value,dict):
    if(isinstance(value,str)) :
        splitted_string=value.split(",")
        val=0
        for string in splitted_string :
            val +=compute_comparison(string,dict)
        return val
    else :
        return 0


def computescore(itemdf, titlesdict, tagsdict, attribdict):
    items_ids=itemdf["id"]
    itemdf=itemdf.drop("id",axis=1)
    columns_names=itemdf.columns
    for colunm in columns_names :
        if(colunm == "tags") :
            itemdf[colunm]=itemdf[colunm].map(lambda x: compute_comparison_string(x,tagsdict))
        elif(colunm == "title") :
            itemdf[colunm] = itemdf[colunm].map(lambda x: compute_comparison_string(x, titlesdict))
        else :
            element_dict= attribdict[colunm]
            itemdf[colunm]=itemdf[colunm].map(lambda x: compute_comparison(x,element_dict),na_action=None)
    sum_series=itemdf.sum(axis=1)
    dictionary=dict(zip(items_ids.values,sum_series.values))
    return dictionary


def getitemsid(item_indexes, dataset):
    return dataset.loc[item_indexes].id


# Main code of the script
total_tic = dt.now()
top_pop = [1053452, 2778525, 1244196, 1386412, 657183]
with open("test.csv", "w") as f:
    f.write("user_id,recommended_items\n")
    for user in user_ids:
        tic = dt.now()
        titles, tags, attrib = createdictionary(user, interactions, items)
        if len(attrib) > 0:
            items_score = computescore(available_items, titles, tags, attrib)  # se questo è un dizionario in fprma {itemid: score} basta de-commentare le righe sotto ed è fatta
            # Sort by score
            sorted_id = sorted(items_score.items(), key=operator.itemgetter(1), reverse=True)
            # Save the first 5 elements
            recommended_ids = []
            for elem in sorted_id[:5]:
                recommended_ids.append(elem[0])
        else:
            print("USER {} has no ratings, recommendations done based on jobroles".format(user))
            user_row = users[users.user_id == user]
            u_jobroles = get_jobroles(user_row)
            recommended_ids = recommend_no_ratings(u_jobroles,available_items)
            print("\tjobroles: {}".format(u_jobroles))
            print("\trecommandations: {}".format(recommended_ids))
            i = 0
            while len(recommended_ids) < 5:
                recommended_ids.append(top_pop[i])
                i += 1
        f.write("{},{}\n".format(user, ' '.join(str(e) for e in recommended_ids)))
        print("User {} computed in {}".format(user, dt.now() - tic))

print("Process ended after {}".format(dt.now()-total_tic))
