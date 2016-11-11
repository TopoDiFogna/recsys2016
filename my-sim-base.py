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


def get_user_preferred_data(items):
    career_level = items.groupby(by="career_level", as_index=False).size().sort_values(ascending=False).index[0]
    discipline_id = items.groupby(by="discipline_id", as_index=False).size().sort_values(ascending=False).index[0]
    industry_id = items.groupby(by="industry_id", as_index=False).size().sort_values(ascending=False).index[0]
    country = items.groupby(by="country", as_index=False).size().sort_values(ascending=False).index[0]
    region = 0  # Defaults to this to be safe in case of disasters in the data
    if country == "de":
        region = items.groupby(by=["country", "region"], as_index=False).size().sort_values(ascending=False).index[0][1]
    employment = items.groupby(by="employment", as_index=False).size().sort_values(ascending=False).index[0]
    return career_level, discipline_id, industry_id, country, region, employment


# Filters the items for the user profile
def recommend(career_level, discipline_id, industry_id, country, region, employment):
    filtered_items = items[(items.career_level == career_level) &
                           (items.discipline_id == discipline_id) &
                           (items.discipline_id == discipline_id) &
                           (items.industry_id == industry_id) &
                           (items.country == country) &
                           (items.employment == employment) &
                           (items.active_during_test == 1)]  # IMPORTANTE!
    # If region is meaningful we use it
    if region != 0:
        filtered_items = filtered_items[filtered_items.region == region]

    print(filtered_items)


# groupby_cols = ["career_level", "discipline_id", "industry_id", "country", "region", "employment"]
for user in user_ids:
    ratings = getuserratings(user)
    rated_items = pd.DataFrame(columns=items.columns).astype(np.int32)  # Empty DataFrame to store rated items
    for rating in ratings:
        if not ratings.size == 0:
            rated_items = rated_items.append(items[items.id == rating], ignore_index=True)
    if not rated_items.empty:
        # Saving preferred data
        mr_career_level, mr_discipline_id, mr_industry_id, mr_country, mr_region, mr_employment = get_user_preferred_data(rated_items)
        # Printing preferred data
        print("USER {}:\n\tcareer level: {}\n\tdiscipline id: {}\n\tindustry id: {}\n\tcountry: {}\n\tregion: {}\n\temployment: {}"
               .format(user, mr_career_level, mr_discipline_id, mr_industry_id, mr_country, mr_region, mr_employment))
        # Do the recommendations
        #recommend(mr_career_level, mr_discipline_id, mr_industry_id, mr_country, mr_region, mr_employment)
    else:
        pass  # TODO aggiungere quelli vuoti
    break
