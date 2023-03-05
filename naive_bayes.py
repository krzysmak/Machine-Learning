"""
Task: Given the training data in the table below (Tennis data with some numerical attributes),
without using sklearn library, predict the class of the following new example using Naïve Bayes classification:

outlook=overcast, temperature=60, humidity=62, windy=false
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_openml
import math

# Load the nominal WEATHER dataset from https://www.openml.org/
weather = datasets.fetch_openml(name='weather', version=2)
print('Features:',   weather.feature_names)
print('Target(s):',  weather.target_names)
print('Categories:', weather.categories)

df = pd.DataFrame( np.c_[weather.data, weather.target],
                   columns=np.append(weather.feature_names, weather.target_names) )

# Modified task: Do not take 'windy' parameter into the account.
def pCondForClassValue(column, value, classColumn, classValue):
  cFalse, cTrue = df.groupby([classColumn])[classColumn].count()
  count = df.loc[df[column] == value].groupby(classColumn)[classColumn].count()
  false = count['no'] if 'no' in count else 0
  true = count['yes'] if 'yes' in count else 0
  if classValue:
    return true / cTrue
  else:
    return false / cFalse

def pCond(column, value, classColumn):
  return (pCondForClassValue(column, value, classColumn, False), 
          pCondForClassValue(column, value, classColumn, True))

def pCondRealValuesForClassValue(column, value, classColumn, classValue):
  data = df.groupby([classColumn])[column]
  mean = data.mean()
  variance = data.var()
  if classValue:
    mean = mean['yes'] if 'yes' in mean else 0
    variance = variance['yes'] if 'yes' in variance else 0
  else:
    mean = mean['no'] if 'no' in mean else 0
    variance = variance['no'] if 'no' in variance else 0
  exponent = math.exp(-((value - mean)**2 / (2 * variance**2 )))
  return (1 / (math.sqrt(2 * math.pi * variance))) * exponent

def pCondRealValues(column, value, classColumn):
  return (pCondRealValuesForClassValue(column, value, classColumn, False), 
          pCondRealValuesForClassValue(column, value, classColumn, True))

def predictClass(outlook, temperature, humidity):
  cFalse, cTrue = df.groupby(['play'])['play'].count()
  outlookNotPlay, outlookPlay = pCond('outlook', outlook, 'play')
  temperatureNotPlay, temperaturePlay = pCondRealValues('temperature', temperature, 'play')
  humidityNotPlay, humidityPlay = pCondRealValues('humidity', humidity, 'play')
  pYes = outlookPlay * temperaturePlay * humidityPlay * cTrue
  pNo = outlookNotPlay * temperatureNotPlay * humidityNotPlay * cFalse
  if pYes > pNo:
    return 'yes'
  else:
    return 'no'

print("Will a person play? " + predictClass('overcast', 60, 62))
