import pandas as pd
import string
import numpy as np

items = pd.read_table("data\item_profile.csv",header=0,sep="\t")
tagSeries=pd.Series()
print(items[items["id"] == 2542195].title)
items.fillna("0", inplace=True)
value=0
for tagRow in items.title :
    for tag in tagRow.split(",") :
        if(not(tag in tagSeries.index)) and not(tag == "0") :
            tagSeries.set_value(tag,value)
            value+=1
with open("title_matrix.csv", "w") as f:
    tagSeries.to_csv(f, sep=',')

print(tagSeries)