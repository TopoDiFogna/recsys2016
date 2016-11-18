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


def computescore(itemdf, titlesdic, tagsdict, attribdict):
    title_score = 0
    tags_score = 0
    attrib_score = 0
    pass


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
            # # Sort by score
            # sorted_id = sorted(item_score.items(), key=operator.itemgetter(1), reverse=True)
            # # Save the first 5 elements
            # recommended_ids = []
            # for elem in sorted_id[:5]:
            #     recommended_ids.append(getitemsid(elem[0], available_items))
        else:
            print("USER {} has no ratings".format(user))
            pass  # TODO aggiungere utenti senza ratings

        print("User {} computed in {}".format(user, dt.now() - tic))
        break

print("Process ended after {}".format(dt.now()-total_tic))
