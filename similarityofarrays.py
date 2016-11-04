import pandas as pd
import numpy as np

items= pd.read_table("data/item_profile.csv",header=0,sep="\t")

s=0

for first_row in items.tags:
    print(first_row)
    if not (isinstance(first_row, float)):
        for second_row in items.tags:
            if not (first_row != second_row and isinstance(second_row, float)):
                a=np.array(first_row.split(","))
                b=np.array(second_row.split(","))
                s = len(np.intersect1d(a,b))/len(np.union1d(a,b))
            else:
                pass
    else:
        pass

