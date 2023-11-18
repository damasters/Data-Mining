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

train_file_name = folder_path + 'yelp_train.csv'
review_file = 'review_train.json'
user_file = 'user.json'
business_file = 'business.json'

def load_train_val(path, bool):
    lines = sc.textFile(path)
    header = lines.first()
    if bool == True:
      return lines.filter(lambda x: x != header).map(lambda x: x.split(",")).map(lambda x: (x[1], x[0], float(x[2])))
    else:
      return lines.filter(lambda x: x != header).map(lambda x: x.split(",")).map(lambda x: (x[1], x[0]))

def avg_ratings(data):
    return data.map(lambda x: (x[0], x[2])).groupByKey().mapValues(lambda vals: sum(vals) / len(vals))

def business_user_ratings(data):
    return data.map(lambda row: (row[0], {row[1]: row[2]})).reduceByKey(lambda a, b: {**a, **b})

start_time = time.time()
train_data = load_train_val(train_file_name, True)
val_data = load_train_val(test_file_name, False)
business_user = train_data.map(lambda row: (row[0], {row[1]})).reduceByKey(lambda a, b: a | b).collectAsMap()
user_business = train_data.map(lambda row: (row[1], {row[0]})).reduceByKey(lambda a, b: a | b).collectAsMap()
business_avg = train_data.map(lambda x: (x[0], x[2])).groupByKey().mapValues(lambda vals: sum(vals) / len(vals)).collectAsMap()
total_rating_bus = business_avg.values()
overall_bus_avg = sum(total_rating_bus) / len(total_rating_bus)
user_avg = train_data.map(lambda x: (x[1], x[2])).groupByKey().mapValues(lambda vals: sum(vals) / len(vals)).collectAsMap()
total_rating_user = user_avg.values()
overall_user_avg = sum(total_rating_user) / len(total_rating_user)
business_user_rating = train_data.map(lambda row: (row[0], {row[1]: row[2]})).reduceByKey(lambda a, b: {**a, **b}).collectAsMap()

def compute_pearson_correlation(ratings_bus, ratings_business, intersection):
    common_users = len(intersection)
    avg_bus = sum(ratings_bus[user] for user in intersection) / common_users
    avg_business = sum(ratings_business[user] for user in intersection) / common_users

    norm_bus = [ratings_bus[user] - avg_bus for user in intersection]
    norm_business = [ratings_business[user] - avg_business for user in intersection]
    numerator = sum([ri * rj for ri, rj in zip(norm_bus, norm_business)])
    denominator = sqrt(sum([val ** 2 for val in norm_bus])) * sqrt(sum([val ** 2 for val in norm_business]))
    return 0 if denominator == 0 else numerator / denominator

def few_corated_items(bus_avg_or_val, business_avg_or_val):
  difference = abs(bus_avg_or_val - business_avg_or_val)
  pear = 1 - (difference / 5.0)
  return pear

def prediction(busid, userid):

    if userid not in user_business:
        return overall_user_avg
    if busid not in business_user:
        return overall_bus_avg

    user_few_inter = len(user_business[userid]) < 7
    bus_few_inter = len(business_user[busid]) < 7

    if user_few_inter:
      if bus_few_inter:
        return overall_bus_avg
      else:
        return overall_user_avg
    if bus_few_inter:
      return overall_bus_avg

    similar_items = []
    big_n = 20
    default_pred = 3.7
    weight_cf = 0.5

    for business in user_business[userid]:

      ratings_bus = business_user_rating.get(busid)
      ratings_business = business_user_rating.get(business)

      corated = [k for k in ratings_bus if k in ratings_business]
      n = len(corated)

      if n <= 1:
        pearson = few_corated_items(business_avg[busid], business_avg[business])

      elif n == 2:
        pearson_one = few_corated_items(ratings_bus[corated[0]], ratings_business[corated[0]])
        pearson_two = few_corated_items(ratings_bus[corated[1]], ratings_business[corated[1]])
        pearson = 0.5 * (pearson_one + pearson_two)

      else:
        pearson = compute_pearson_correlation(ratings_bus, ratings_business, corated)

      if pearson < weight_cf:
        continue

      rating_business = business_user_rating[business].get(userid)
      similar_items.append((pearson, rating_business))
    similar_items = heapq.nlargest(big_n, similar_items, key=lambda x: x[0])
    numerator = sum(pear * rate for pear, rate in similar_items)
    denominator = sum(abs(pear) for pear, rate in similar_items)
    return default_pred if denominator == 0 else numerator / denominator

ib_user_bus_preds = val_data.map(lambda x: (x[1], x[0], prediction(x[0], x[1])))
ib_predictions = predictions = ib_user_bus_preds.map(lambda x: x[2]).collect()

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

load_train = load_csv_file(train_file_name)
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
mb_predictions = xgb_model.predict(x_val_test)

zip_pred = zip(ib_predictions, mb_predictions)
zip_val_preds = list(zip(val_user_business, zip_pred))

alpha = 0.05
final_scores = []
with open(output_file_name, "w") as output_file:
  output_file.write("user_id, business_id, prediction\n")
  for (user, business), (pred_ib, pred_mb) in zip_val_preds:
    final_score = (alpha * pred_ib) + ((1-alpha)* pred_mb)
    final_scores.append(final_score)
    output_file.write(f"{user},{business},{final_score}\n")
end_time = time.time()
print('Duration: ', end_time - start_time)
