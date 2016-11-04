import pandas as pd
import numpy as np

tags=pd.read_table("tag_matrix.csv", sep=",")
titles=pd.read_table("title_matrix.csv", sep=",")
items = pd.read_table("data\item_profile.csv",header=0,sep="\t")

for index in titles.index.values :
    items[index]=""

for index in tags.index.values :
    items[index]=""

with open("big_item.csv", "w") as f:
    items.to_csv(f, sep=',')

print(items[1])
