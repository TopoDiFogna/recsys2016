def getinteractionusers(usersdf, interactionsdf):
    interactionsusers = interactionsdf["user_id"].unique()
    usersdf = usersdf[usersdf["user_id"].isin(interactionsusers)]
    return usersdf


def compute_comparison(value, dictionary):
    if value in dictionary:
        return dictionary[value]
    else:
        return 0


def compute_comparison_array(value, denom, dictionary):
    if value in dictionary:
        return dictionary[value] / denom
    else:
        return 0


def compute_comparison_string(value, dictionary):
    if isinstance(value, str):
        splitted_string = value.split(",")
        max_denom = max(len(splitted_string), len(dictionary))
        summation = 0
        for string in splitted_string:
            summation += compute_comparison_array(string, max_denom, dictionary)
        return summation
    else:
        return 0


def computenoratingssimilarity(userdf, jobrolesdict, attribdict, edudict):
    items_ids = userdf["user_id"]
    userdf = userdf.drop("user_id", axis=1)
    columns_names = userdf.columns
    for colunm in columns_names:
        if colunm == "jobroles":
            if 0 in jobrolesdict:
                userdf[colunm] = 0
            else:
                userdf[colunm] = userdf[colunm].map(lambda x: compute_comparison_string(x, jobrolesdict))
        elif colunm == "edu_fieldofstudies":
            if 0 in edudict:
                userdf[colunm] = 0
            else:
                userdf[colunm] = userdf[colunm].map(lambda x: compute_comparison_string(x, edudict))
        else:
            element_dict = attribdict[colunm]
            userdf[colunm] = userdf[colunm].map(lambda x: compute_comparison(x, element_dict), na_action=None)
    sum_series = userdf.sum(axis=1)
    dictionary = dict(zip(items_ids.values, sum_series.values))
    return dictionary


def create_dictionary_user(userwow):
    userwow = userwow.drop("user_id", axis=1)
    columns_names = userwow.columns
    attrdict = {}
    jobrolesdict = {}
    edudict = {}
    for colunm in columns_names:
        if colunm == "jobroles":
            value = userwow[colunm].values.item(0)
            if not (value == "0"):
                splitted_string = value.split(",")
                for string in splitted_string:
                    jobrolesdict[string] = 1
            else:
                jobrolesdict[0] = 0
        elif colunm == "edu_fieldofstudies":
            value = userwow[colunm].values.item(0)
            if not (value == "0"):
                splitted_string = value.split(",")
                for string in splitted_string:
                    edudict[string] = 1
            else:
                edudict[0] = 0
        else:
            if userwow[colunm].values.item(0) == 0:
                attrdict[colunm] = dict({userwow[colunm].values.item(0): 0})
            else:
                attrdict[colunm] = dict({userwow[colunm].values.item(0): 1})
    return attrdict, jobrolesdict, edudict


def compute_recommendations(ids_dict, interactionsdf, items_filtered):
    items_filtered_ids = items_filtered["id"].values
    recommended_ids = []
    for u_id in ids_dict:
        if len(recommended_ids) < 5:
            selectedinteractions = interactionsdf[interactionsdf["user_id"] == u_id]
            selectedinteractions = selectedinteractions[interactionsdf["item_id"].isin(items_filtered_ids)].sort_values(
                "interaction_type", ascending=False)
            selectedinteractions = selectedinteractions["item_id"].unique()
            for entry in selectedinteractions:
                recommended_ids.append(entry)
        else:
            break
    return recommended_ids[:5]
