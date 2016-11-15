import numpy as np
import pandas as pd
from collections import Counter
import operator
from datetime import datetime as dt

# Loading Data
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
samples = pd.read_csv("data/sample_submission.csv", header=0)
items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
users = pd.read_table("data/user_profile.csv", sep="\t", header=0)
# End loading data

# Prepocessing data
user_ids = samples.user_id.values
items.fillna(value="0", inplace=True)
# End of prepocessing data

#
# def getitemsid(item_indexes,dataset):
#     return dataset.loc[item_indexes].id.values
#
#
# # Gets the ratings a user has performed dropping the duplicates and keeping the highest
# def getuserratings(userid):
#     sampleinteractions = interactions.loc[interactions['user_id'] == userid].reset_index().drop("index", 1).drop(
#         "created_at", 1).drop("user_id", 1)
#     sampleinteractions = sampleinteractions.groupby(by='item_id', as_index=False).apply(
#         lambda x: x.ix[x.interaction_type.idxmax()])
#     if sampleinteractions.empty:
#         return np.array([])
#     else:
#         return sampleinteractions.item_id.values
#
#
# def get_user_preferred_data(items_rated):
#     career_level = items_rated.groupby(by="career_level", as_index=False).size().sort_values(ascending=False).index[0]
#     discipline_id = items_rated.groupby(by="discipline_id", as_index=False).size().sort_values(ascending=False).index[0]
#     industry_id = items_rated.groupby(by="industry_id", as_index=False).size().sort_values(ascending=False).index[0]
#     country = items_rated.groupby(by="country", as_index=False).size().sort_values(ascending=False).index[0]
#     region = 0  # Defaults to this to be safe in case of disasters in the data
#     if country == "de":
#         region = items_rated.groupby(by=["country", "region"], as_index=False).size().sort_values(ascending=False) \
#             .index[0][1]
#     employment = items_rated.groupby(by="employment", as_index=False).size().sort_values(ascending=False).index[0]
#     return career_level, discipline_id, industry_id, country, region, employment
#
#
# def get_user_profile(user_id):
#     user_row = users[users.user_id == user_id]
#     career_level = user_row.career_level
#     discipline_id = user_row.discipline_id
#     industry_id = user_row.industry_id
#     country = user_row.country
#     region = user_row.region
#     jobroles = user_row.jobroles.split(',')
#     return career_level, discipline_id, industry_id, country, region, jobroles
#
#
# def most_common_attribute(array):
#     count = Counter(array)
#     return [tag[0] for tag in count.most_common()]
#
#
# def get_titles_ordered(items_rated):
#     titles = np.array([])
#     for title in items_rated.title.values:
#         titles = np.append(titles, title.split(","))
#     most_common = most_common_attribute(titles)
#     return most_common
#
#
# def get_tags_ordered(items_rated):
#     tags = np.array([])
#     for tag in items_rated.tags.values:
#         tags = np.append(tags, tag.split(","))
#     most_common = most_common_attribute(tags)
#     return most_common
#
#
# def recommend_no_ratings(career_level, discipline_id, industry_id, country, region, jobroles):
#     filtered_items = items[(items.career_level == career_level) &  # TODO check for 0
#                            (items.discipline_id == discipline_id) &
#                            (items.discipline_id == discipline_id) &
#                            (items.industry_id == industry_id) &
#                            (items.country == country) &
#                            (items.active_during_test == 1)]  # IMPORTANTE!
#     # If region is meaningful we use it
#     if region != 0:
#         filtered_items = filtered_items[filtered_items.region == region]
#     recommended_id = {}
#     for index, row in filtered_items.iterrows():
#         if list(set(jobroles) & set(row.tags.split(','))) and list(set(jobroles) & set(row.title.split(','))):
#             recommended_id[row.id] = len(list(set(jobroles) & set(row.tags.split(','))))  # TODO non so se sia giusto
#     sorted_id = sorted(recommended_id.items(), key=operator.itemgetter(1), reverse=True)
#     recommendations = []
#     for elem in sorted_id[:5]:
#         recommendations.append(elem[0])
#     return recommendations
#
#
# def get_tags_intersection(row, tags):
#     if list(set(tags) & set(row.split(','))):
#         element = len(list(set(tags) & set(row.split(','))))
#         return element
#     else:
#         return 0  # funzione che fa l'intersezione
#
#
# # Filters the items for the user profile
# def recommend(career_level, title, discipline_id, industry_id, country, region, employment, tags):
#     # filtered_items = items[(items.career_level == career_level) &  # TODO check for 0
#     #                        (items.discipline_id == discipline_id) &
#     #                        (items.discipline_id == discipline_id) &      # TODO ho tolto i filtri, lavora solo sui tag
#     #                        (items.industry_id == industry_id) &
#     #                        (items.country == country) &
#     #                        (items.employment == employment) &  # TODO check for 0
#     #                        (items.active_during_test == 1)]  # IMPORTANTE!
#     filtered_items = items[items.active_during_test == 1].reset_index()
#     # If region is meaningful we use it
#     # if region != 0:
#     #     filtered_items = filtered_items[filtered_items.region == region]
#
#     element = filtered_items['tags'].apply((lambda x: get_tags_intersection(x, tags))) #instruzione che applica la funzione per ogni riga
#     # for index, row in filtered_items.iterrows():
#     #     if list(set(tags) & set(row.tags.split(','))) and list(set(title) & set(row.title.split(','))):
#     #         recommended_id[row.id] = len(list(set(tags) & set(row.tags.split(','))))
#     recommended_id = element.values
#     top_rated_items_id = recommended_id.argsort()[-5:][::-1]
#     recommendations = []
#     for elem in top_rated_items_id :
#         recommendations.append(getitemsid(elem, filtered_items))
#     # sorted_id = sorted(recommended_id.items(), key=operator.itemgetter(1), reverse=True)
#     # recommendations = []
#     # for elem in sorted_id[:5]:
#     #     recommendations.append(elem[0])
#
#     return recommendations

# total_tic = dt.now()
# top_pop = [1053452, 2778525, 1244196, 1386412, 657183]
# with open("test.csv", "w") as f:
#     f.write("user_id,recommended_items\n")
#     for user in user_ids:
#         tic = dt.now()
#         ratings = getuserratings(user)
#         rated_items = pd.DataFrame(columns=items.columns).astype(np.int32)  # Empty DataFrame to store rated items
#         for rating in ratings:
#             if not ratings.size == 0:
#                 rated_items = rated_items.append(items[items.id == rating], ignore_index=True)
#
#         if not rated_items.empty:
#             # Saving preferred data
#             mr_career_level, mr_discipline_id, mr_industry_id, mr_country, mr_region, mr_employment = \
#                 get_user_preferred_data(rated_items)
#             mr_titles = get_titles_ordered(rated_items)
#             mr_tags = get_tags_ordered(rated_items)
#             # Printing preferred data
#             print("USER {}:".format(user))
#             print("\ttitles: {}".format(mr_titles))
#             print("\tcareer level: {}".format(mr_career_level))
#             print("\tdiscipline id: {}".format(mr_discipline_id))
#             print("\tindustry id: {}".format(mr_industry_id))
#             print("\tcountry: {}".format(mr_country))
#             print("\tregion: {}".format(mr_region))
#             print("\temployment: {}".format(mr_employment))
#             print("\ttags: {}".format(mr_tags))
#             # Do the recommendations
#             recommended_ids = recommend(mr_career_level, mr_titles, mr_discipline_id, mr_industry_id, mr_country,
#                                         mr_region, mr_employment, mr_tags)
#             i = 0
#             while len(recommended_ids) < 5:
#                 print("\trecommendations overrided: {}".format(recommended_ids))
#                 recommended_ids.append(top_pop[i])
#                 i += 1
#             print("\trecommendations: {}".format(recommended_ids))
#         else:
#             print("User {} don't have any interaction".format(user))
#             print("Recommending top pop: {}".format(top_pop))
#             recommended_ids = top_pop
#             # u_career_level, u_discipline_id, u_industry_id, u_country, u_region, u_jobroles = get_user_profile(user)
#             # recommended_ids = recommend_no_ratings(u_career_level, u_discipline_id, u_industry_id, u_country, u_region,
#             #                                        u_jobroles)
#         f.write("{},{}\n".format(user, ' '.join(str(e) for e in recommended_ids)))
#         print("User {} computed in {}".format(user, dt.now()-tic))
# print("Process ended after {}".format(dt.now()-total_tic))


# Gets the ratings a user has performed dropping the duplicates and keeping the highest
def getuserratings(userid):
    sampleinteractions = interactions[interactions['user_id'] == userid].reset_index().drop("index", 1).drop(
        "created_at", 1).drop("user_id", 1)
    sampleinteractions = sampleinteractions.groupby(by='item_id', as_index=False).apply(
        lambda x: x.ix[x.interaction_type.idxmax()])
    if sampleinteractions.empty:
        return np.array([])
    else:
        return sampleinteractions.item_id.values


def get_tags_ordered(items_rated):
    tags = np.array([])
    for tag in items_rated.tags.values:
        tags = np.append(tags, tag.split(","))
    most_common = most_common_attribute(tags)
    return most_common


def most_common_attribute(array):
    count = Counter(array)
    return [tag[0] for tag in count.most_common()]


def recommend(rated_user_items, rated_tags):
    available_items = items[items.active_during_test == 1]
    for item_id in rated_user_items.id:
        available_items = available_items[available_items.id != item_id]

    print(available_items)

total_tic = dt.now()
top_pop = [1053452, 2778525, 1244196, 1386412, 657183]
with open("test.csv", "w") as f:
    f.write("user_id,recommended_items\n")
    for user in user_ids:
        tic = dt.now()
        ratings = getuserratings(user)
        rated_items = pd.DataFrame(columns=items.columns).astype(np.int32)
        for rating in ratings:
            if not ratings.size == 0:  # User has reted something
                rated_items = rated_items.append(items[items.id == rating], ignore_index=True)
        if not rated_items.empty:  # User has some ratngs
            rated_items_tags = get_tags_ordered(rated_items)
            print("USER: {}".format(user))
            print("\trated items: {}".format(rated_items.id.values.tolist()))
            print("\ttags: {}".format(rated_items_tags))
            recommended_ids = recommend(rated_items, rated_items_tags)
        else:  # User has no ratings
            recommended_ids = top_pop
        #f.write("{},{}\n".format(user, ' '.join(str(e) for e in recommended_ids)))
        print("User {} computed in {}".format(user, dt.now() - tic))
        break
print("Process ended after {}".format(dt.now()-total_tic))
