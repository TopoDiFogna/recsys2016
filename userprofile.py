from tfidf import *


# Gets the ratings a user has performed dropping the duplicates and keeping the highest
def getuserratings(userid, interactionsdf):
    sampleinteractions = interactionsdf[interactionsdf['user_id'] == userid].reset_index().drop("index", 1).drop(
        "created_at", 1).drop("user_id", 1)
    if sampleinteractions.empty:
        return {}
    else:
        return sampleinteractions.item_id.values


# Returns a dataframe containing the info of the given item
def getitemprofile(itemid, itemsdf):
    item_profile = itemsdf[itemsdf['id'] == itemid]
    return item_profile.drop(["id", "created_at", "active_during_test"], axis=1).squeeze()


def get_index_form_id(itemid, itemsdf):
    index = itemsdf[itemsdf["id"] == itemid].index.values[0]
    return index


# Returns 3 different dictionaries for the interactions of the user, every attribute is weighted by number of ratings
def createdictionary(userid, interactionsdf, itemsdf, title_matrix, tag_matrix, tagdf, titledf):
    titledict = {}
    tagsdict = {}
    attributesdict = {}  # This is a nested dictionary!
    user_ratings = getuserratings(userid, interactionsdf)
    for rated_item in user_ratings:
        item_profile = getitemprofile(rated_item, itemsdf)
        index_item = get_index_form_id(rated_item, itemsdf)
        for key in item_profile.index.values:
            if key != "title" and key != "tags":
                if key not in attributesdict:
                    attributesdict[key] = dict({item_profile[key]: 1})
                else:
                    if item_profile[key] not in attributesdict[key]:
                        attributesdict[key][item_profile[key]] = 1
                    else:
                        attributesdict[key][item_profile[key]] += 1
            elif key == "title":
                titles = item_profile.title.split(',')
                if not (titles[0] == "0"):
                    for title in titles:
                        if title not in titledict:
                            titledict[title] = tf_idfcomputing(title_matrix, index_item, titledf.loc[int(title)])
                        else:
                            titledict[title] += tf_idfcomputing(title_matrix, index_item, titledf.loc[int(title)])
            elif key == "tags":
                tags = item_profile.tags.split(',')
                if not (tags[0] == "0"):
                    for tag in tags:
                        if tag not in tagsdict:
                            tagsdict[tag] = tf_idfcomputing(tag_matrix, index_item, tagdf.loc[int(tag)])
                        else:
                            tagsdict[tag] += tf_idfcomputing(tag_matrix, index_item, tagdf.loc[int(tag)])
    return titledict, tagsdict, attributesdict
