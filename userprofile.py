# TODO RICORDATI DI INSERIRE QUESTI NEL TUO SCRIPT
# TODO RICORDATI DI INSERIRE QUESTI NEL TUO SCRIPT
# TODO RICORDATI DI INSERIRE QUESTI NEL TUO SCRIPT
# TODO RICORDATI DI INSERIRE QUESTI NEL TUO SCRIPT
# interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
# items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
# items.fillna(value="0", inplace=True)

# Per importare questo modulo e usare le funzioni qui dentro basta usare:
# from userprifile import createdictionary
# ti serve solo quella funzione, le altre sono di supporto


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


# Returns 3 different dictionaries for the interactions of the user, every attribute is weighted by number of ratings
def createdictionary(userid, interactionsdf, itemsdf):
    titledict = {}
    tagsdict = {}
    attributesdict = {}  # This is a nested dictionary!
    user_ratings = getuserratings(userid, interactionsdf)
    for rated_item in user_ratings:
        item_profile = getitemprofile(rated_item, itemsdf)
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
                for title in titles:
                    if title not in titledict:
                        titledict[title] = 1
                    else:
                        titledict[title] += 1
            elif key == "tags":
                tags = item_profile.tags.split(',')
                for tag in tags:
                    if tag not in tagsdict:
                        tagsdict[tag] = 1
                    else:
                        tagsdict[tag] += 1
    return titledict, tagsdict, attributesdict
