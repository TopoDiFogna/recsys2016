import pandas as pd
import numpy as np
from datetime import datetime as dt
import operator

from pandas.lib import item_from_zerodim

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
available_items = items[items.active_during_test == 1].drop(["active_during_test", "created_at", "latitude", "longitude"], axis=1)


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


def compute_comparison(value, dictionary,base):
    if value in dictionary:
        if base == 0 :
            return dictionary[value]
        else :
            if dictionary[value] == 0 :
                return 0
            else:
                return  base ** dictionary[value]
    else:
        return 0


def compute_comparison_string(value, dictionary,base):
    if isinstance(value, str):
        splitted_string = value.split(",")
        summation = 0
        for string in splitted_string:
            summation += compute_comparison(string, dictionary,base)
        return summation
    else:
        return 0


def computescore(itemdf, titlesdict, tagsdict, attribdict, alreadyclickeditems):
    items_ids = itemdf["id"]
    itemdf = itemdf.drop("id", axis=1)
    columns_names = itemdf.columns
    for colunm in columns_names:
        if colunm == "tags":
            itemdf[colunm] = itemdf[colunm].map(lambda x: compute_comparison_string(x, tagsdict,0))
        elif colunm == "title":
            itemdf[colunm] = itemdf[colunm].map(lambda x: compute_comparison_string(x, titlesdict,0))
        else:
            element_dict = attribdict[colunm]
            #itemdf[colunm] = itemdf[colunm].map(lambda x: compute_comparison(x, element_dict,0), na_action=None)
            #questa riga computa i ratings anche per gli altri valori, l'ho commentata per fare speed_up
    #sum_series = itemdf.sum(axis=1)
    sum_series = itemdf["tags"] + itemdf["title"]
    dictionary = dict(zip(items_ids.values, sum_series.values))
    for item in alreadyclickeditems:
        if item in dictionary:
            dictionary[item] = 0
    return dictionary


def getitemsid(item_indexes, dataset):
    return dataset.loc[item_indexes].id

#questo metodo server per riordinare le recommendations con lo stesso ordine
def orderRatings (sorteddict, tagsdict,titlesdict,attribdict, availableitems):
    orderedratings=[]
    while len(orderedratings) <5 :
        maxvalue = sorteddict[0][1]
        equalids=[]
        for elem in sorteddict :
            if elem[1] == maxvalue :
                equalids.append(elem[0])
            if elem[1] < maxvalue :
                break
        sorteddict=sorteddict[len(equalids):]
        if(len(equalids)>1) :

            max_tag_value=max(tagsdict.values())
            max_title_value=max(titlesdict.values())
            max_attr_value=0
            for elem in attribdict :
                max_temp=max(attribdict[elem].values())
                if(max_temp>max_attr_value) :
                    max_attr_value = max_temp

            item_selected=available_items[available_items.id.isin(equalids)]
            ids=item_selected["id"]
            item_selected=item_selected.drop("id", axis=1)

            # potresti avere dei problemi con questa base perchè uso una valutazione di tipo esponenziale
            #ho aggiunto questo controllo per evitare l'overflow di float
            if(max_attr_value >= 308 or max_title_value >=308 or max_tag_value >=308) :
                base = 5
            else :
                base =10


            for colunm in item_selected.columns:
                if colunm == "tags":
                    item_selected[colunm] = item_selected[colunm].map(lambda x: compute_comparison_string(x, tagsdict, base))
                elif colunm == "title":
                    item_selected[colunm] = item_selected[colunm].map(lambda x: compute_comparison_string(x, titlesdict, base))
                else:
                    element_dict = attribdict[colunm]
                    #item_selected[colunm] = item_selected[colunm].map(lambda x: compute_comparison(x, element_dict, base), na_action=None)
                    #questa l'ho tolta per speed_up come nell'altro metodo
            #sum_series=item_selected.sum(axis=1).sort_values(ascending=False)
            sum_series = item_selected["tags"] + item_selected["title"]
            sum_series = sum_series.sort_values(ascending=False)
            sum_indexes=sum_series.index
            for index in sum_indexes :
                orderedratings.append(ids[index])
        else :
            orderedratings.append(equalids[0])
    return orderedratings[:5]


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
            recommended_ids=orderRatings(sorted_id,tags,titles,attrib,available_items)
            # Save the first 5 elements
            # for elem in sorted_id[:5]:
            #     recommended_ids.append(elem[0])
            print(recommended_ids)
        else:
            print("USER {} has no ratings, recommendations done based on jobroles".format(user))
            user_row = users[users.user_id == user]
            u_jobroles = get_jobroles(user_row)
            recommended_ids = recommend_no_ratings(u_jobroles, available_items)
            print("\tjobroles: {}".format(u_jobroles))
            print("\trecommandations: {}".format(recommended_ids))
            i = 0
            while len(recommended_ids) < 5:
                recommended_ids.append(top_pop[i])
                i += 1
            # attrdict, jobdict, edudict = create_dictionary_user(user_row)
            # returndict = computenoratingssimilarity(interaction_user_df, jobdict, attrdict, edudict)
            # sorted_id = sorted(returndict.items(), key=operator.itemgetter(1), reverse=True)
            # simil_users = []
            # for elem in sorted_id[:10]:
            #     simil_users.append(elem[0])
            # recommended_ids=compute_recommendations(simil_users, interactions, available_items)
        f.write("{},{}\n".format(user, ' '.join(str(e) for e in recommended_ids)))
        print("User {} computed in {}".format(user, dt.now() - tic))

print("Process ended after {}".format(dt.now() - total_tic))
