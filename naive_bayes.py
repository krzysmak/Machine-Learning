"""
Task: Given the training data in the table below (Tennis data with some numerical attributes),
without using sklearn library, predict the class of the following new example using Naïve Bayes classification:

outlook=overcast, temperature=60, humidity=62, windy=false
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_openml

# Load the nominal WEATHER dataset from https://www.openml.org/
weather = datasets.fetch_openml(name='weather', version=2)
print('Features:',   weather.feature_names)
print('Target(s):',  weather.target_names)
print('Categories:', weather.categories)

df = pd.DataFrame( np.c_[weather.data, weather.target],
                   columns=np.append(weather.feature_names, weather.target_names) )