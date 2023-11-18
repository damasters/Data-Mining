import shutil
from pyspark import SparkContext, SparkConf
import itertools
from itertools import groupby
import os
from itertools import combinations, chain
import json
import sys
import time
import math
import datetime
import csv
from datetime import datetime
from collections import defaultdict
from math import sqrt
import random
import heapq
import xgboost as xgb
import numpy as np

if 'sc' in globals():
    sc.stop()
    
spark_conf = SparkConf().setAppName('Assignment3').setMaster('local')
sc = SparkContext(conf=spark_conf)
sc.setLogLevel('ERROR')

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

train_file = 'yelp_train.csv'
review_file = 'review_train.json'
user_file = 'user.json'
business_file = 'business.json'

def load_csv_file(filepath):
    data = sc.textFile(filepath)
    header = data.first()
    return data.filter(lambda line: line != header).map(lambda line: line.split(","))

def load_json_file(filepath):
    return sc.textFile(filepath).map(lambda line: json.loads(line))

def extract_attributes(user_id, business_id):
    business_features = bus_stars_per_business.get(business_id, False)
    user_features = user_rc_as_elite.get(user_id, False)
    reviews_user_features = reviews_avg_useful_votes_user.get(user_id, False)
    default_value = np.nan

    if business_features != False:
      stars = business_features[0]
      bus_review_count = business_features[1]
    else:
      stars = default_value
      bus_review_count = default_value

    if user_features != False:
      user_review_count = user_features[0]
      avg_stars = user_features[1]
      elite = user_features[2]
      if elite == True:
        elite = 1.0
      else:
        elite = 0.0
    else:
      user_review_count = default_value
      avg_stars = default_value
      elite = default_value

    if reviews_user_features != False:
      useful = reviews_user_features[0]
      funny = reviews_user_features[1]
      cool = reviews_user_features[2]
    else:
      useful = default_value
      funny = default_value
      cool = default_value

    return (stars, bus_review_count, user_review_count, avg_stars, elite, useful, funny, cool)

start_time = time.time() 
load_train = load_csv_file(folder_path + train_file)
load_val_test = load_csv_file(test_file_name)
reviews = load_json_file(folder_path + review_file) 
users = load_json_file(folder_path+ user_file)
businesses = load_json_file(folder_path + business_file)

bus_stars_per_business = businesses.map(lambda x: (x['business_id'], (x['stars'], x['review_count']))).mapValues(list).collectAsMap()
user_rc_as_elite = users.map(lambda x: (x['user_id'], (float(x['review_count']), float(x['average_stars']), bool(x['elite'])))).collectAsMap()
reviews_avg_useful_votes_user = reviews.map(lambda x: (x['user_id'], (float(x['useful']), float(x['funny']), float(x['cool']), 1)))\
                    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])).mapValues(lambda x: (x[0] / x[3], x[1] / x[3], x[2] / x[3])).collectAsMap()
reviews_avg_useful_votes_business = reviews.map(lambda x: (x['business_id'], (float(x['useful']), float(x['funny']), float(x['cool']), 1)))\
                    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])).mapValues(lambda x: (x[0] / x[3], x[1] / x[3], x[2] / x[3])).collectAsMap()

train = load_train.map(lambda x: (extract_attributes(x[0], x[1]), float(x[2]))).collectAsMap()

x_train = np.array([list(key) for key in train.keys()], dtype='float32')
y_train = np.array(list(train.values()), dtype='float32')

val_user_business = load_val_test.map(lambda x: (x[0], x[1])).collect()
val_test = load_val_test.map(lambda x: (extract_attributes(x[0], x[1]))).collect()
x_val_test = np.array(list(val_test), dtype='float32')

xgb_model = xgb.XGBRegressor()
xgb_model.fit(x_train, y_train)
predictions = xgb_model.predict(x_val_test)

with open(output_file_name, "w") as output_file:
  output_file.write("user_id, business_id, prediction\n")
  for (user, business), pred in zip(val_user_business, predictions):
    output_file.write(f"{user},{business},{pred}\n")

end_time = time.time()
print('Duration: ', end_time - start_time)