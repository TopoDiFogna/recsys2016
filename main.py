import pandas as pd
import numpy as np
import csv

items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
users = pd.read_table("data/user_profile.csv", sep="\t", header=0)
interactions = pd.read_table("data/interactions.csv", sep="\t", header=0)
sample = pd.read_table("data/sample_submission.csv", sep=",", header=0)

colFuckingName =  sample.columns[0]
sample = sample.rename(columns = {colFuckingName:'user_id'})
sampleIds = sample.user_id
usersids = users.user_id.sort_values().reset_index(drop=True)

print(sampleIds)

availableItems = items[items.active_during_test == 1]

clickedJobs = interactions[interactions.interaction_type ==1]

mostFiveClicked = clickedJobs.groupby("item_id").size().sort_values(ascending=False)[:5]
print(mostFiveClicked)

columns = ['user_id', 'recommended_items']
df= pd.DataFrame(index=range(10000), columns=columns)
df['user_id']=sampleIds
df['recommended_items']="1053452 2778525 1244196 1386412 657183"
#print(df)
with open("recommendations.csv", "w") as f:
    df.to_csv(f,sep=',', index=False)



