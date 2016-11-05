import pandas as pd
from  datetime import datetime as dt
import numpy as np

items = pd.read_table("data\item_profile.csv",header=0,sep="\t")
tic=dt.now()
result= pd.DataFrame(index=range(len(items.index.values)), columns=items.columns.values)

for index,row in items.iterrows() :

    row["career_level"] /= 6
    row["discipline_id"] /= 23
    row["industry_id"] /=23

    country=row["country"]
    if country == "de" :
        row["country"] = 0.33
    elif country == "at" :
        row["country"] = 0.66
    elif country == "ch" :
        row["country"] = 0.99
    elif country == "non_dach":
        row["country"] = 0

    row["region"] /= 16

    latitude=row["latitude"]
    if isinstance(latitude, np.float64):
        latitude = 0
    elif latitude <0 :
        latitude = 90 + abs(latitude)
    row["latitude"] = latitude/180

    longitude = row["longitude"]
    if isinstance(longitude,np.float64):
        longitude = 0
    elif longitude < 0:
        longitude = 180 + abs(longitude)
    row["longitude"] = longitude / 360

    row["region"] /= 5

    row["employment"] /= 5

    result.loc[index]=row

result.fillna(0)

with open("normalized.csv", "w") as f:
    result.to_csv(f, sep=',',index=False)
