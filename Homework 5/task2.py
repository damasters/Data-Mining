import shutil
import itertools
from pyspark import SparkContext, SparkConf
from itertools import groupby
import os
from itertools import combinations, chain
from itertools import permutations
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
import copy
import binascii
from blackbox import BlackBox

input_filename = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_filename = sys.argv[4]

def myhashs(s):
  user_id_int = int(binascii.hexlify(s.encode('utf8')),16)
  m = 877
  a = random.sample(range(1, m+1), 100)
  b = random.sample(range(1, m+1), 100)
  output_list = []
  for num in range(len(a)):
    a_i = a[num]
    b_i = b[num]
    func = ((a_i * user_id_int) + b_i) % m
    output_list.append(func)
  return output_list

def trailing_zeros(n):
    s = bin(n)
    return len(s) - len(s.rstrip('0'))

def flajolet_martin(users, num_hash_funcs):
    max_zeros = [0] * num_hash_funcs
    for user in users:
        hash_values = myhashs(user)
        for i, hash_val in enumerate(hash_values):
            max_zeros[i] = max(max_zeros[i], trailing_zeros(hash_val))
    est = [2**zero for zero in max_zeros]
    avg_estimate = sum(est) // len(est)
    return avg_estimate

start_time = time.time()
time_gt_est = []

for num_ask in range(num_of_asks):
    users = BlackBox().ask(input_filename, stream_size)
    unique_users_count = len(set(users))  # Ground truth
    estimate = flajolet_martin(users, 100)  # Assuming 100 hash functions
    time_gt_est.append((num_ask, unique_users_count, estimate))

#gt_sum = sum(j for i,j,k in time_gt_est)
#estimates_sum = sum(k for i,j,k in time_gt_est)
#print(estimates_sum/gt_sum)

with open(output_filename, 'w') as output:
    csvwrite = csv.writer(output)
    csvwrite.writerow(['Time', 'Ground Truth', 'Estimation'])
    for tup in time_gt_est:
        csvwrite.writerow([tup[0], tup[1], tup[2]])

end_time = time.time()
print('Duration: '+str(end_time - start_time))