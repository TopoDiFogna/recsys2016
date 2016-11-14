import numpy as np
import pandas as pd
from collections import Counter
import operator
from datetime import datetime as dt

# Loading Data
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
samples = pd.read_csv("data/sample_submission.csv", header=0)
items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
# End loading data

# Prepocessing data
user_ids = samples.user_id.values
items.fillna(value="0", inplace=True)
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


def get_user_preferred_data(items_rated):
    career_level = items_rated.groupby(by="career_level", as_index=False).size().sort_values(ascending=False).index[0]
    discipline_id = items_rated.groupby(by="discipline_id", as_index=False).size().sort_values(ascending=False).index[0]
    industry_id = items_rated.groupby(by="industry_id", as_index=False).size().sort_values(ascending=False).index[0]
    country = items_rated.groupby(by="country", as_index=False).size().sort_values(ascending=False).index[0]
    region = 0  # Defaults to this to be safe in case of disasters in the data
    if country == "de":
        region = items_rated.groupby(by=["country", "region"], as_index=False).size().sort_values(ascending=False) \
            .index[0][1]
    employment = items_rated.groupby(by="employment", as_index=False).size().sort_values(ascending=False).index[0]
    return career_level, discipline_id, industry_id, country, region, employment


def most_common_attribute(array):
    count = Counter(array)
    return [tag[0] for tag in count.most_common()]


def get_titles_ordered(items_rated):
    titles = np.array([])
    for title in items_rated.title.values:
        titles = np.append(titles, title.split(","))
    most_common = most_common_attribute(titles)
    return most_common


def get_tags_ordered(items_rated):
    tags = np.array([])
    for tag in items_rated.tags.values:
        tags = np.append(tags, tag.split(","))
    most_common = most_common_attribute(tags)
    return most_common


# Filters the items for the user profile
def recommend(career_level, title, discipline_id, industry_id, country, region, employment, tags):
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

    recommended_id = {}
    for index, row in filtered_items.iterrows():
        if list(set(tags) & set(row.tags.split(','))) and list(set(title) & set(row.title.split(','))):
            recommended_id[row.id] = len(list(set(tags) & set(row.tags.split(','))))
    sorted_id = sorted(recommended_id.items(), key=operator.itemgetter(1), reverse=True)
    recommendations = []
    for elem in sorted_id[:5]:
        recommendations.append(elem[0])

    return recommendations

total_tic = dt.now()
top_pop = [1053452, 2778525, 1244196, 1386412, 657183]
with open("test.csv", "w") as f:
    for user in user_ids:
        tic = dt.now()
        ratings = getuserratings(user)
        rated_items = pd.DataFrame(columns=items.columns).astype(np.int32)  # Empty DataFrame to store rated items
        for rating in ratings:
            if not ratings.size == 0:
                rated_items = rated_items.append(items[items.id == rating], ignore_index=True)

        if not rated_items.empty:
            # Saving preferred data
            mr_career_level, mr_discipline_id, mr_industry_id, mr_country, mr_region, mr_employment = \
                get_user_preferred_data(rated_items)
            mr_titles = get_titles_ordered(rated_items)
            mr_tags = get_tags_ordered(rated_items)
            # Printing preferred data
            print("USER {}:".format(user))
            print("\ttitles: {}".format(mr_titles))
            print("\tcareer level: {}".format(mr_career_level))
            print("\tdiscipline id: {}".format(mr_discipline_id))
            print("\tindustry id: {}".format(mr_industry_id))
            print("\tcountry: {}".format(mr_country))
            print("\tregion: {}".format(mr_region))
            print("\temployment: {}".format(mr_employment))
            print("\ttags: {}".format(mr_tags))
            # Do the recommendations
            recommended_ids = recommend(mr_career_level, mr_titles, mr_discipline_id, mr_industry_id, mr_country,
                                        mr_region, mr_employment, mr_tags)
            i = 0
            while len(recommended_ids) < 5:
                print("\trecommendations overrided: {}".format(recommended_ids))
                recommended_ids.append(top_pop[i])
                i += 1
            print("\trecommendations: {}".format(recommended_ids))
        else:
            print("No interactions for user: {}".format(user))
            recommended_ids = [1053452, 2778525, 1244196, 1386412, 657183]  # TODO fix empty user
        f.write("{},{}\n".format(user, ' '.join(str(e) for e in recommended_ids)))
        print("User {} computed in {}".format(user, dt.now()-tic))
print("Process ended after {}".format(dt.now()-total_tic))
