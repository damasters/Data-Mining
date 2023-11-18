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

def reservoir_sampling(str_usrs, res, seq_num):
    for user in str_usrs:
        seq_num += 1
        if len(res) < 100:
            res.append(user)
        else:
            if random.random() < 100 / seq_num:
                replace_index = random.randint(0, 99)
                res[replace_index] = user
    return seq_num

if __name__ == '__main__':
    input_filename = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_filename = sys.argv[4]

    start_time = time.time()
    random.seed(553)

    res = []
    seq_num = 0

    with open(output_filename, 'w', newline='') as output:
        csvwrite = csv.writer(output)
        csvwrite.writerow(['seqnum', '0_id', '20_id', '40_id', '60_id', '80_id'])
        for num_ask in range(num_of_asks):
            str_usrs = BlackBox().ask(input_filename, stream_size)
            seq_num = reservoir_sampling(str_usrs, res, seq_num)
            if seq_num % 100 == 0:
                selected_users = [res[i] for i in [0, 20, 40, 60, 80]]
                csvwrite.writerow([seq_num] + selected_users)
    end_time = time.time()
    print('Duration: '+str(end_time - start_time))