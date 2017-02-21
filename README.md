# Recommender Systems 2016

University project for the course of __Recommender Systems__ 2016 hold by Prof. __Paolo Cremonesi__ at __Politecnico di Milano__

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

In order to run the project you'll need Python version 3 and above ([LINK](https://www.python.org/downloads/))
You will also need this pip packages:

```
numpy==1.12.0+mkl
pandas==0.19.2
scipy==0.18.1
```

### Installing

After having installed Python just run

```
pip install numpy pandas scipy
```

When everything is set up just run
```
python tf-idf-ratings.py
```

You should get something like

```
Loading data...
Data loaded in 0:00:01.081037
Prepocessing Data...
Ended preprocessing in 0:00:00.117112
Starting recommending!

User 285
[749025, 827566, 1808563, 1763102, 1965165]
User 285 computed in 0:00:04.406224
...
```

The output will be stored in the file `test.csv` in the csv format like:
```
user1,rec1,rec2,rec3,rec4,rec5
user2,rec1,rec2,rec3,rec4,rec5
```

###Troubleshooting

If you miss some of the file the files in the directory `precomputedData` you can generate them doing the following:

###### userRatingSimilarity_IP.npz
* Uncomment lines 9-14 in `utils\cfutils.py`
* Comment line 18 in `utils\cfutils.py`
* Run function `create_user_rating_matrix_similarity` in `utils\cfutils.py`

###### user_rating_matrix.npz
* Uncomment lines 9-14 in `utils\cfutils.py`
* Comment line 18 in `utils\cfutils.py`
* Run function `create_user_rating_matrix`  in `utils\cfutils.py`

### Branches
The master branch holdes the code that scored most in the competition, every other branch is named by the algorithm used to create the recommendations

### Coding style

Whenever possible the code tries to follow recommendations specified by [PEP8](https://www.python.org/dev/peps/pep-0008/)

###Data and Submissions
The data can be downloaded from the platform [Kaggle](https://inclass.kaggle.com/c/recommender-system-2016-challenge-polimi)

All the submission done to the platform [Kaggle](https://inclass.kaggle.com/c/recommender-system-2016-challenge-polimi) can be found in the folder submission tagged with the date

## Authors

* **Paolo Paterna**
* **Lorenzo Zoia**
