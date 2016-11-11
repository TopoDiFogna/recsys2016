import numpy as np
import pandas as pd

# Loading Data
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
samples = pd.read_csv("data/sample_submission.csv", header=0)
items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
# End loading data

# Prepocessing data
user_ids = samples.user_id.values
items.fillna(value=0, inplace=True)
# End of prepocessing data


# Gets the ratings a user has performed dropping the duplicates and keeping the highest
def getuserratings(userid):
    sampleinteractions = interactions.loc[interactions['user_id'] == userid].reset_index().drop("index", 1).drop(
        "created_at", 1).drop("user_id", 1)
    sampleinteractions = sampleinteractions.groupby(by='item_id', as_index=False).apply(
        lambda x: x.ix[x.interaction_type.idxmax()])
    if sampleinteractions.empty:
        return np.array([])
    else:
        return sampleinteractions.item_id.values

print(getuserratings(48478))

groupby_cols = ["career_level", "discipline_id", "industry_id", "employment"]
for user in user_ids:
    print("USER: {}".format(user))
    ratings = getuserratings(user)
    rated_items = pd.DataFrame(columns=items.columns).astype(np.int32)
    for rating in ratings:
        if not ratings.size == 0:
            print("ITEM: {}".format(rating))
            rated_items = rated_items.append(items[items.id == rating], ignore_index=True)
        # most_rated = rated_items.groupby(by=groupby_cols).size().sort_values(ascending=False).head(n=1)
        mr_career_level, mr_discipline_id, mr_industry_id, mr_employment = rated_items.groupby(by=groupby_cols, as_index=False, sort=False).size().sort_values(ascending=False).index[0]
