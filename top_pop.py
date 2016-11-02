import pandas as pd
import numpy as np
import scipy.sparse as sps
import argparse
import logging
from datetime import datetime as dt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

# items = pd.read_table("data/item_profile.csv", sep="\t", header=0)
# users = pd.read_table("data/user_profile.csv", sep="\t", header=0)
# sample = pd.read_table("data/sample_submission.csv", sep=",", header=0)
#
# availableItems = items[items.active_during_test == 1]
# clickedJobs = interactions[interactions.interaction_type == 1]

# colFuckingName = sample.columns[0]
# sample = sample.rename(columns={colFuckingName: 'user_id'})
# sampleIds = sample.user_id
# usersids = users.user_id.sort_values().reset_index(drop=True)
#
# print(sampleIds)
#
# availableItems = items[items.active_during_test == 1]
#
#
# mostFiveClicked = clickedJobs.groupby("item_id").size().sort_values(ascending=False)[:5]
# print(mostFiveClicked)
#
# columns = ['user_id', 'recommended_items']
# df = pd.DataFrame(index=range(10000), columns=columns)
# df['user_id'] = sampleIds
# df['recommended_items'] = "1053452 2778525 1244196 1386412 657183"
# # print(df)
# with open("recommendations.csv", "w") as f:
#     df.to_csv(f, sep=',', index=False)

# C = []
# for r in zip(interactions['user_id'], interactions['item_id']):
#     C.append((r[0], r[1]))


def read_dataset(path, header=None, columns=None, user_key='user_id', item_key='item_id'):
    data = pd.read_table(path, header=header, names=columns)
    logger.info('Columns: {}'.format(data.columns.values))
    # build user and item maps (and reverse maps)
    # this is used to map ids to indexes starting from 0 to nitems (or nusers)
    items = data[item_key].unique()
    users = data[user_key].unique()
    item_to_idx = pd.Series(data=np.arange(len(items)), index=items)
    user_to_idx = pd.Series(data=np.arange(len(users)), index=users)
    idx_to_item = pd.Series(index=item_to_idx.data, data=item_to_idx.index)
    idx_to_user = pd.Series(index=user_to_idx.data, data=user_to_idx.index)
    #  map ids to indices
    data['item_idx'] = item_to_idx[data[item_key].values].values
    data['user_idx'] = user_to_idx[data[user_key].values].values
    return data, idx_to_user, idx_to_item


def holdout_split(data, perc=0.8, seed=1234):
    # set the random seed
    rng = np.random.RandomState(seed)
    # shuffle data
    nratings = data.shape[0]  # numbers of rows
    shuffle_idx = rng.permutation(nratings)
    train_size = int(nratings * perc)
    # split data according to the shuffled index and the holdout size
    train_split = data.ix[shuffle_idx[:train_size]]
    test_split = data.ix[shuffle_idx[train_size:]]
    return train_split, test_split


def df_to_csr(df, nrows, ncols, user_key='user_idx', item_key='item_idx', rating_key='interaction_type'):
    rows = df[user_key].values
    columns = df[item_key].values
    ratings = df[rating_key].values
    shape = (nrows, ncols)
    # using the 4th constructor of csr_matrix
    # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    return sps.csr_matrix((ratings, (rows, columns)), shape=shape)


class TopPop(object):
    """Top Popular recommender"""

    def __init__(self):
        super(TopPop, self).__init__()

    def fit(self, train):
        if isinstance(train, sps.csr_matrix):
            # convert to csc matrix for faster column-wise sum
            train_csc = train.tocsc()
        else:
            train_csc = train
        item_pop = (train_csc > 0).sum(axis=0)  # this command returns a numpy.matrix of size (1, nitems)
        item_pop = np.asarray(item_pop).squeeze()  # necessary to convert it into a numpy.array of size (nitems)
        self.pop = np.argsort(item_pop)[::-1]  # sorts the array by specifying the indexes of the elements (reversed)

    def recommend(self, profile, k=None, exclude_seen=True):
        unseen_mask = np.in1d(self.pop, profile, assume_unique=True, invert=True)
        return self.pop[unseen_mask][:k]


# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--holdout_perc', type=float, default=0.8)
parser.add_argument('--header', type=int, default=None)
parser.add_argument('--columns', type=str, default=None)
parser.add_argument('--sep', type=str, default='\t')
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--rating_key', type=str, default='interaction_type')
parser.add_argument('--rnd_seed', type=int, default=1234)
args = parser.parse_args()

# read the dataset
logger.info('Reading {}'.format(args.dataset))
dataset, idx_to_user, idx_to_item = read_dataset(
    args.dataset,
    header=args.header,
    columns=args.columns,
    item_key=args.item_key,
    user_key=args.user_key)
nusers, nitems = len(idx_to_user), len(idx_to_item)
logger.info('The dataset has {} users and {} items'.format(nusers, nitems))

# compute the holdout split
logger.info('Computing the {:.0f}% holdout split'.format(args.holdout_perc * 100))
train_df, test_df = holdout_split(dataset, perc=args.holdout_perc, seed=args.rnd_seed)
train = df_to_csr(train_df, nrows=nusers, ncols=nitems)
test = df_to_csr(test_df, nrows=nusers, ncols=nitems)

# top-popular recommender
logger.info('Building the top-popular recommender')
recommender = TopPop()
tic = dt.now()  # saves current time
logger.info('Training started')
recommender.fit(train)
logger.info('Training completed built in {}'.format(dt.now() - tic))

# ranking quality evaluation
roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
at = 5
neval = 0
data = pd.read_table("data\sample_submission.csv", header=0)
print(data)
columns = ['user_id', 'recommended_items']
df = pd.DataFrame(index=range(10000), columns=columns)
index=0
for test_user in range(nusers):
    user_id = dataset[dataset.user_idx == test_user].user_id.unique().item(0)
    if any(data["user_id"] == user_id) :
        user_profile = train[test_user].indices# items that the user has interacted with?
        recommended_items = recommender.recommend(user_profile, k=at, exclude_seen=True)
        for key in range(len(recommended_items)):
            item_id=dataset[dataset.item_idx == recommended_items[key]].item_id.unique().item(0)
            recommended_items[key]=item_id
        df.loc[index]=[user_id,np.array_str(recommended_items)]
        index=index + 1
        print(index)

for test_user in range(10000) :
    user_id=data.get_value(test_user,"user_id")
    if not(any(df["user_id"] == user_id)) :
        user_profile = train[0].indices
        recommended_items = recommender.recommend(user_profile, k=at, exclude_seen=True)
        for key in range(len(recommended_items)):
            item_id=dataset[dataset.item_idx == recommended_items[key]].item_id.unique().item(0)
            recommended_items[key]=item_id
        df.loc[index] = [user_id, np.array_str(recommended_items)]
        index = index + 1
        print(index)

with open("recommendations_new.csv", "w") as f:
    df.to_csv(f, sep=',', index=False)