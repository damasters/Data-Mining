# -*- coding: utf-8 -*-

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
from datetime import datetime
from collections import defaultdict

if 'sc' in globals():
    sc.stop()
spark_conf = SparkConf().setAppName('Assignment2').setMaster('local')
sc = SparkContext(conf=spark_conf)

case_num_filepath = int(sys.argv[1])
support_filepath = int(sys.argv[2])
input_filepath = sys.argv[3]
output_filepath = sys.argv[4]

def freq_items_generation(baskets, support, candidates):
  item_counts = {item: sum(1 for b in baskets if item.issubset(b)) for item in candidates}
  return [item for item, count in item_counts.items() if count >= support]

def cand_items_generation(freq_items):
    sorted_tuples = [tuple(sorted(item)) for item in freq_items]
    if not sorted_tuples:
      return set()
    len_itemset = len(sorted_tuples[0])
    len_prefix = len_itemset - 1
    
    prefix_dict = {}
    for item in sorted_tuples:
        prefix = item[:len_prefix]
        suffix = item[len_prefix]
        # prefix = item[:-1]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(suffix)

    new_candidates = set()
    for prefix, suffix in prefix_dict.items():
      for combo in combinations(suffix, 2):
        new_itemset = tuple(sorted(prefix + combo))
        new_candidates.add(frozenset(new_itemset))

    return new_candidates

def new_apriori_algo(partition, support, first_candidates):
  baskets = [frozenset(basket) for basket in partition]
  new_candidates = [frozenset(candidate) for candidate in first_candidates]
  freq_items_list = []
  while new_candidates:
    freq_items = freq_items_generation(baskets, support, new_candidates)
    freq_items_list.extend(freq_items)
    new_candidates = cand_items_generation(freq_items)
    if not new_candidates:
      break

  return freq_items_list

def format_items(itemsets):
    sorted_items = [tuple(sorted(itemset)) for itemset in itemsets]
    sorted_items = sorted(sorted_items, key=lambda x: (len(x), x))
    grouped_items = groupby(sorted_items, key=lambda x: len(x))
    return '\n\n'.join(','.join(f"('{item[0]}')" if len(item) == 1 else str(item) for item in group) for x, group in grouped_items)

def son_algorithm(case_number, support, input_fpath, output_fpath): #add input file path and output file path
  start_time = time.time()
  big_data = sc.textFile(input_filepath)
  header = big_data.first()
  data_wo_header = big_data.filter(lambda x: x != header)
  if case_number == 1:
    qualified_b = data_wo_header.map(lambda line: line.split(",")).map(lambda x: (x[0], x[1]))\
             .groupByKey().mapValues(set).map(lambda x: list(x[1]))
  elif case_number == 2:
    qualified_b = data_wo_header.map(lambda line: line.split(",")).map(lambda x: (x[1], x[0]))\
             .groupByKey().mapValues(set).map(lambda x: list(x[1]))
  num_partitions = qualified_b.getNumPartitions()
  local_sup = math.ceil(support/float(num_partitions))
  single_cands = qualified_b.flatMap(lambda x: [(item, 1) for item in x]).reduceByKey(lambda x,y: x+y)
  freq_singles = single_cands.filter(lambda x: x[1] >= local_sup).map(lambda x: ({x[0]})).collect()
  phase_1 = qualified_b.mapPartitions(lambda partition: new_apriori_algo(partition, local_sup, freq_singles))\
            .map(lambda x: tuple(x)).collect()
  phase_2 = qualified_b.flatMap(lambda basket: [(candidate, 1) for candidate in phase_1 \
                    if set(candidate).issubset(basket)]).reduceByKey(lambda a, b: a + b) \
                    .filter(lambda x: x[1] >= support).collect()

  with open(output_filepath, 'w') as f:
    f.write("Candidates:\n")
    f.write(format_items(phase_1))
    f.write("\n\nFrequent Itemsets:\n")
    f.write(format_items([item[0] for item in phase_2]))

  end_time = time.time()
  print("Duration:", end_time - start_time)

son_algorithm(case_num_filepath, support_filepath, input_filepath, output_filepath)