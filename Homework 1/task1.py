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

blackbox = BlackBox()
m = 69997  
num_hashes = 100  
a = random.sample(range(1, m), num_hashes)
b = random.sample(range(1, m), num_hashes)

def myhashs(s):
  user_id_int = int(binascii.hexlify(s.encode('utf8')),16)
  output_list = []
  for num in range(len(a)):
    a_i = a[num]
    b_i = b[num]
    func = ((a_i * user_id_int) + b_i) % m
    output_list.append(func)
  return output_list

def bloom_filter_func(user, bf, user_exists, hashes):
  is_fp = True
  for hash_val in hashes:
    if bf[hash_val] == 0:
      is_fp = False
      break

  if is_fp and user not in user_exists:
    user_exists.add(user)
    for hash_val in hashes:
      bf[hash_val] = 1
    return True

  user_exists.add(user)
  for hash_val in hashes:
    bf[hash_val] = 1
  return False

start_time = time.time()
m = 69997
bloom_filter = [0] * m
all_users = set()
fprs = []

for num_ask in range(num_of_asks):
  users = blackbox.ask(input_filename, stream_size)
  fp_count = 0
  for user in users:
      hashes = myhashs(user)
      if bloom_filter_func(user, bloom_filter, all_users, hashes):
          fp_count += 1
  fpr = fp_count / stream_size
  fprs.append((num_ask, fpr))

with open(output_filename, 'w') as output:
  csvwrite = csv.writer(output)
  csvwrite.writerow(['Time', 'FPR'])
  for tup in fprs:
    csvwrite.writerow([tup[0], tup[1]])
    
end_time = time.time()
print('Duration: '+str(end_time - start_time))