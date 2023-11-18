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

if 'sc' in globals():
    sc.stop()
spark_conf = SparkConf().setAppName('Assignment3').setMaster('local')
sc = SparkContext(conf=spark_conf)
sc.setLogLevel('ERROR')


train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]


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

results = val_data.map(lambda x: (x[1], x[0], prediction(x[0], x[1]))).collect()

with open(output_file_name, "w") as f:
    f.write("user_id, business_id, prediction \n")
    for user, bus, pred in results:
      f.write(f"{user},{bus},{pred}\n")

end_time = time.time()
print('Duration: ', end_time - start_time)