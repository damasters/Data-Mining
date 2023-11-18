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

spark_conf = SparkConf().setAppName('Assignment3').setMaster('local')
sc = SparkContext(conf=spark_conf)
sc.setLogLevel('ERROR')

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]

start_time = time.time()
data = sc.textFile(input_file_name)
header = data.first()
data_wo_header = data.filter(lambda x: x != header)
parsed_data = data_wo_header.map(lambda x: x.split(","))
unique_users = parsed_data.map(lambda x: x[0]).distinct()

all_bus_user = parsed_data.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collect()
create_user_ints = dict(zip(unique_users.collect(), range(1, unique_users.count()+1)))

def convert_user_ids(user_ints, all_businesses_and_users):
  user_ints_set = set(user_ints.keys())
  new_dict = {}
  for a, b in all_businesses_and_users:
    user_ids = b
    for i in user_ids:
      # list(user_ints.keys())
      if i in user_ints_set:
        new_user_id = create_user_ints[i]
        if a in new_dict:
          new_dict[a].append(new_user_id)
        else:
          new_dict[a] = [new_user_id]

  return new_dict

def min_hash(all_bus_and_users, combined_ab):
    signatures_dict = {}
    for business_id, userids in all_bus_and_users:
      signatures = []
      for a, b in combined_ab:
        minhash_vals = []
        for userid in userids:
          hash_value = (((a * create_user_ints[userid] + b) % p) % m)
          minhash_vals.append(hash_value)
        signatures.append(min(minhash_vals))
      signatures_dict[business_id] = signatures
    return signatures_dict

def generate_buckets(signature_matrix, b, r):
    buckets = {}
    large_buckets = {}
    for business_id, signatures in signature_matrix.items():
        for i in range(b):
            band = tuple(signatures[i*r:(i+1)*r])
            if band not in buckets:
                buckets[band] = []
            buckets[band].append(business_id)
    for band, business_ids in buckets.items():
        if len(business_ids) > 1:
            large_buckets[band] = business_ids
    return large_buckets

def candidates_jaccard(buckets, users_to_ints_dict):
  candidates = set()
  final = []
  for business_ids in buckets.values(): 
    two_item_combo = combinations(sorted(business_ids), 2)
    for combo in two_item_combo:
      candidates.add(combo)
  for business1, business2 in candidates: 
    values_bus_one = users_to_ints_dict[business1]
    values_bus_two = users_to_ints_dict[business2]
    size_intersection = len(set(values_bus_one).intersection(set(values_bus_two)))
    size_union = len(set(values_bus_one).union(set(values_bus_two)))
    jaccard_sim = size_intersection / size_union
    if jaccard_sim >= 0.5:
      final.append((business1, business2, jaccard_sim))
  sorted_final = sorted(final, key=lambda x: (x[0], x[1]))
  return sorted_final

new_bus_user = convert_user_ids(create_user_ints, all_bus_user)
num_hash_functions = 150
m = unique_users.count()
p = 350117
a_values = random.sample(range(1, m), num_hash_functions)
b_values = random.sample(range(1, m), num_hash_functions)

combined_ab = list(zip(a_values, b_values))
sig_matrix = min_hash(all_bus_user, combined_ab)
b = 50 
r = 3 
buckets_gt_one = generate_buckets(sig_matrix, b, r)
result = candidates_jaccard(buckets_gt_one, new_bus_user)

with open(output_file_name, 'w') as output:
  csv_writer = csv.writer(output)
  csv_file_header = ('business_id_1', 'business_id_2', 'similarity')
  csv_writer.writerow(csv_file_header)
  for line in result:
    csv_writer.writerow(line)

end_time = time.time()
print('Duration: ', end_time - start_time)