import pandas as pd
from datetime import datetime as dt
from wightmanagement import getuseritemweights, createdictionary
import operator

# Loading Data
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
samples = pd.read_csv("data/sample_submission.csv", header=0)
users = pd.read_table("data/user_profile.csv", sep="\t", header=0)
# End loading data

# Prepocessing data
samples_user_ids = samples.user_id.values
items.fillna(value="0", inplace=True)
users.fillna(value="0", inplace=True)
available_items = items[items.active_during_test == 1].drop(["active_during_test", "created_at"], axis=1)
# End of prepocessing data


# Gets the ratings a user has performed dropping the duplicates and keeping the highest
def getuserratings(userid, interactionsdf):
    sampleinteractions = interactionsdf[interactionsdf['user_id'] == userid].reset_index().drop("index", 1).drop(
        "created_at", 1).drop("user_id", 1)
    return sampleinteractions


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

    sum_series = itemdf["tags"] + itemdf["title"] + itemdf["industry_id"] + itemdf["discipline_id"] + itemdf["country"]
    dictionary = dict(zip(items_ids.values, sum_series.values))
    for item in alreadyclickeditems:
        if item in dictionary:
            dictionary[item] = 0
    return dictionary


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

            max_tag_value = max(tagsdict.values())
            max_title_value = max(titlesdict.values())
            max_attr_value = 0
            for elem in attribdict:
                max_temp = max(attribdict[elem].values())
                if max_temp > max_attr_value:
                    max_attr_value = max_temp

            item_selected = availableitems[availableitems.id.isin(equalids)]
            ids = item_selected["id"]
            item_selected = item_selected.drop("id", axis=1)

            # potresti avere dei problemi con questa base perchè uso una valutazione di tipo esponenziale
            # ho aggiunto questo controllo per evitare l'overflow di float
            if max_attr_value >= 308 or max_title_value >= 308 or max_tag_value >= 308:
                base = 5
            else:
                base = 10

            for colunm in item_selected.columns:
                if colunm == "tags":
                    item_selected[colunm] = item_selected[colunm].map(
                        lambda x: compute_comparison_string(x, tagsdict, base))
                elif colunm == "title":
                    item_selected[colunm] = item_selected[colunm].map(
                        lambda x: compute_comparison_string(x, titlesdict, base))

            sum_series = item_selected["tags"] + item_selected["title"]
            sum_series = sum_series.sort_values(ascending=False)
            sum_indexes = sum_series.index
            for index in sum_indexes:
                orderedratings.append(ids[index])
        else:
            orderedratings.append(equalids[0])
    return orderedratings[:5]


def get_jobroles(row):
    jobdict = {}
    for jobrole in row.jobroles.values:
        if jobrole not in jobdict:
            jobdict[jobrole] = 1
        else:
            jobdict[jobrole] += 1
    return jobdict


def recommend_no_ratings(jobroles, attribdict, rnr_available_items):
    items_ids = rnr_available_items["id"]
    # Dict containing {item_index: count}
    rnr_available_items['tags'] = rnr_available_items['tags'].apply(lambda x: compute_comparison_string(x, jobroles, 0))
    rnr_available_items['title'] = rnr_available_items['title'].apply(lambda x: compute_comparison_string(x, jobroles,
                                                                                                          0))
    for k in attribdict:
        element_dict = attribdict[k]
        rnr_available_items[k] = rnr_available_items[k].map(lambda x: compute_comparison(x, element_dict, 0),
                                                            na_action=None)
    sum_series = (rnr_available_items["tags"] + rnr_available_items["title"] + rnr_available_items["industry_id"] +
                  rnr_available_items["discipline_id"] + rnr_available_items["country"])
    dictionary = dict(zip(items_ids.values, sum_series.values))
    # Sort by count
    rnr_sorted_id = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    # Save the first 5 elements
    recommendations = []
    for rnr_elem in rnr_sorted_id[:5]:
        recommendations.append(rnr_elem[0])
    return recommendations


def get_tags_intersection(row, in_tags):
    if list(set(in_tags) & set(row.split(','))):
        count = len(list(set(in_tags) & set(row.split(','))))
        return count
    else:
        return 0


def getuserattributes(row):
    discipline_id = row.discipline_id.values[0]
    industry_id = row.industry_id.values[0]
    country = row.country.values[0]
    dictionary = {"discipline_id": {discipline_id: 1}, "industry_id": {industry_id: 1},
                  "country": {country: 1}}
    return dictionary


# Main code of the script
total_tic = dt.now()
top_pop = [1053452, 2778525, 1244196, 1386412, 657183]
with open("test.csv", "w") as f:
    f.write("user_id,recommended_items\n")
    for user in samples_user_ids:
        tic = dt.now()
        user_ratings = getuserratings(user, interactions)
        user_ratings_weighted = getuseritemweights(user_ratings, False)
        titles, tags, attrib = createdictionary(user_ratings_weighted, items)
        recommended_ids = []
        if len(attrib) > 0:
            items_score = computescore(available_items, titles, tags, attrib, user_ratings.item_id.values)
            sorted_id = sorted(items_score.items(), key=operator.itemgetter(1), reverse=True)
            recommended_ids = order_ratings(sorted_id, tags, titles, attrib, available_items)
            print("User: {}, Recommendations: {}".format(user, recommended_ids))
        else:
            print("USER {} has no ratings, recommendations done based on his profile".format(user))
            user_row = users[users.user_id == user]
            u_jobroles = get_jobroles(user_row)
            u_attrib = getuserattributes(user_row)
            recommended_ids = recommend_no_ratings(u_jobroles, u_attrib, available_items)
            print("User: {}, Recommendations: {}".format(user, recommended_ids))
        f.write("{},{}\n".format(user, ' '.join(str(e) for e in recommended_ids)))
        print("User {} computed in {}".format(user, dt.now() - tic))

print("Process ended after {}".format(dt.now() - total_tic))
