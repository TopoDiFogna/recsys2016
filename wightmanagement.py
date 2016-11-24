# IL FUNZIONAMENTO è : PRIMA CHIAMO getuseritemweights E POI PASSO IL RISULTATO ALLA NUOVA createdictionary

# questa funzione prende in ingresso la porzione di interactions dell'utente e ritorna come output un dizionario
# con i pesi specifici per ogni item. Se gli viene passato come secondo parametro False allora utilizza nel
# calcolo dei pesi anche gli items con rating 2 e 3 se invece gli viene passato True allora usa solo i click


def getuseritemweights(userrating, only_click):
    clickrating = userrating[userrating["interaction_type"] == 1]
    counts = clickrating.groupby('item_id').size()
    maxclicktime = counts.max()
    itemdict = {}
    for item in counts.index:
        itemdict[item] = float("%.4f" % (counts[item] / maxclicktime))+1
    if not only_click:
        preferites = userrating[userrating["interaction_type"] == 2].item_id.unique()
        for u_id in preferites:
            itemdict[u_id] = 10
        applied = userrating[userrating["interaction_type"] == 3].item_id.unique()
        for u_id in applied:
            itemdict[u_id] = 100
    return itemdict


# Returns a dataframe containing the info of the given item
def getitemprofile(itemid, itemsdf):
    item_profile = itemsdf[itemsdf['id'] == itemid]
    return item_profile.drop(["id", "created_at", "active_during_test"], axis=1).squeeze()


# questa funzione invece è il remake di quella vecchia che invece di utilizzare il rating per gli attributi
# usa il nuovo dizionario che si può creare con la funzione sopra

def createdictionary(itemdict, itemsdf):
    titledict = {}
    tagsdict = {}
    attributesdict = {}  # This is a nested dictionary!
    for rated_item in itemdict:
        item_profile = getitemprofile(rated_item, itemsdf)
        for key in item_profile.index.values:
            if key != "title" and key != "tags":
                if key not in attributesdict:
                    attributesdict[key] = dict({item_profile[key]: itemdict[rated_item]})
                else:
                    if item_profile[key] not in attributesdict[key]:
                        attributesdict[key][item_profile[key]] = itemdict[rated_item]
                    else:
                        attributesdict[key][item_profile[key]] += itemdict[rated_item]
            elif key == "title":
                titles = item_profile.title.split(',')
                for title in titles:
                    if title not in titledict:
                        titledict[title] = itemdict[rated_item]
                    else:
                        titledict[title] += itemdict[rated_item]
            elif key == "tags":
                tags = item_profile.tags.split(',')
                for tag in tags:
                    if tag not in tagsdict:
                        tagsdict[tag] = itemdict[rated_item]
                    else:
                        tagsdict[tag] += itemdict[rated_item]
    return titledict, tagsdict, attributesdict
