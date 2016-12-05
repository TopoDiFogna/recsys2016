from tfidf import tf_idfcomputing
import numpy as np


# Gets the ratings a user has performed dropping the duplicates and keeping the highest
def getuserratings(userid, interactionsdf):
    sampleinteractions = interactionsdf[interactionsdf['user_id'] == userid].reset_index().drop("index", 1).drop(
        "created_at", 1).drop("user_id", 1)
    if sampleinteractions.empty:
        return np.array([])
    else:
        return sampleinteractions.item_id.values


# Returns a dataframe containing the info of the given item
def getitemprofile(itemid, itemsdf):
    item_profile = itemsdf[itemsdf['id'] == itemid]
    return item_profile.drop(["id", "created_at", "active_during_test"], axis=1).squeeze()


def get_item_index_form_id(itemid, itemsdf):
    index = itemsdf[itemsdf["id"] == itemid].index.values[0]
    return index


def createdictionary(userid, interactionsdf, itemsdf, title_matrix, tag_matrix, tagdf, titledf):
    titledict = {}
    tagsdict = {}
    user_ratings = getuserratings(userid, interactionsdf)
    for rated_item in user_ratings:
        item_profile = getitemprofile(rated_item, itemsdf)
        index_item = get_item_index_form_id(rated_item, itemsdf)
        for key in item_profile.index.values:
            if key == "title":
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
    return titledict, tagsdict


def getuserprofile(userid, userdf):
    user_profile = userdf[userdf['user_id'] == userid]
    return user_profile.drop(["user_id"], axis=1).squeeze()


def get_user_index_form_id(userid, usersdf):
    index = usersdf[usersdf["user_id"] == userid].index.values[0]
    return index


def createdictionary_noratings(userid, userdf, jobrole_matrix, jobroledf):
    jobroledict = {}
    user_profile = getuserprofile(userid, userdf)
    index_user = get_user_index_form_id(userid, userdf)
    for key in user_profile.index.values:
        if key == "jobroles":
            jobroles = user_profile.jobroles.split(',')
            if not (jobroles[0] == "0"):
                for jobrole in jobroles:
                    if jobrole not in jobroledict:
                        jobroledict[jobrole] = tf_idfcomputing(jobrole_matrix, index_user, jobroledf.loc[int(jobrole)])
                    else:
                        jobroledict[jobrole] += tf_idfcomputing(jobrole_matrix, index_user, jobroledf.loc[int(jobrole)])
    return jobroledict
