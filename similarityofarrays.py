import pandas as pd
import numpy as np


def square(x): return x**2

items= pd.read_table("data/item_profile.csv",header=0,sep="\t")


##TODO prendere in inglesso gli utenti sample
##TODO cercare i lavori con iterazione prendendo quelli con rating massimo
## TODO droppare le colonne di tags e title dal dataFrame

for index, first_row in items.iterrows():#solo quelli cliccati
    for index, second_row in items.iterrows():

        s_tag = 0
        s_title = 0
        i_tag_len = 0
        j_tag_len = 0
        i_title_len = 0
        j_title_len = 0
        s_i_j=0

        if not (isinstance(first_row["tags"], float)):
            a = np.array(first_row["tags"].split(","))
            j_tag_len = len(a)
            if not (first_row != second_row and isinstance(second_row["tags"], float)):
                b = np.array(second_row["tags"].split(","))
                i_tag_len = len(b)
                s_tag = len(np.intersect1d(a, b))
            else:
                pass
        else:
            pass

        if not (isinstance(first_row["title"], float)):
            c = np.array(first_row["title"].split(","))
            j_title_len = len(c)
            if not (first_row != second_row and isinstance(second_row["title"], float)):
                d = np.array(second_row["title"].split(","))
                i_title_len = len(d)
                s_title = len(np.intersect1d(c, d))
            else:
                pass
        else:
            pass
    i_series=first_row.drop(["title","tags"])
    j_series=second_row.drop(["title","tags"])

    result_series=i_series.multiply(j_series)
    result_series["similarity"]= result_series.sum(axis=1)
    similarity=result_series["similarity"].values.item(0)

    i_series=i_series.apply(square)
    j_series=j_series.apply(square)
    i_series["sum"]=i_series.sum(axis=1)
    j_series["sum"] = j_series.sum(axis=1)
    i_sum = i_series["sum"].value.item(0)
    j_sum = j_series["sum"].value.item(0)
    denominator=(i_sum+i_tag_len+i_title_len)**0.5 * (j_sum+j_title_len+j_tag_len)**0.5

    similarity=similarity/denominator + s_title/denominator + s_tag/denominator

##TODO calcolare i rating

##TODO sputare i 5 più alti
