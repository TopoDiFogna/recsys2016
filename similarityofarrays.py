import pandas as pd
import numpy as np

items= pd.read_table("data/item_profile.csv",header=0,sep="\t")

s_tag = 0
s_title = 0

##TODO prendere in inglesso gli utenti sample
##TODO cercare i lavori con iterazione prendendo quelli con rating massimo

for index, first_row in items.iterrows():  #solo quelli cliccati
    for index, second_row in items.iterrows():
        if not (isinstance(first_row["tags"], float)):
            if not (first_row != second_row and isinstance(second_row["tags"], float)):
                a = np.array(first_row["tags"].split(","))
                b = np.array(second_row["tags"].split(","))
                s_tag = len(np.intersect1d(a, b)) / len(a) ** 0.5 * len(b) ** 0.5
            else:
                pass
        else:
            pass

        if not (isinstance(first_row["title"], float)):
            if not (first_row != second_row and isinstance(second_row["title"], float)):
                c = np.array(first_row["title"].split(","))
                d = np.array(second_row["title"].split(","))

                s_title = len(np.intersect1d(c, d)) / len(c) ** 0.5 * len(d) ** 0.5
            else:
                pass
        else:
            pass

    ## TODO droppare le colonne di tags e title dalle Series first_row e second_row
    ## TODO appliccare la formula delle similarities con i due nuovi arrays
    ## TODO sommare le similarities

##TODO calcolare i rating

##TODO sputare i 5 più alti
