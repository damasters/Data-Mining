import shutil
from pyspark import SparkContext, SparkConf
from pyspark.sql.context import SQLContext
from pyspark.sql import Row
from graphframes import GraphFrame
from pyspark.sql.functions import collect_list, size, sort_array, asc
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
spark_conf = SparkConf().setAppName('Assignment2').setMaster('local')
sc = SparkContext(conf=spark_conf)
sc.setLogLevel('ERROR')

threshold_filepath = int(sys.argv[1])
input_filepath = sys.argv[2]
output_filepath = sys.argv[3]

s_t = time.time()
raw_data = sc.textFile(input_filepath)
header = raw_data.first()
data_wo_header = raw_data.filter(lambda x: x != header).map(lambda x: x.split(","))
user_business = data_wo_header.map(lambda x: (x[0], {x[1]})).reduceByKey(lambda a, b: a | b).collectAsMap()
user_distinct = data_wo_header.map(lambda x: x[0]).distinct().collect()

combos = list(combinations(user_distinct, 2))
edges_list = []
nodes_dict = {} 
for tup in combos:
  common_elements = user_business[tup[0]] & user_business[tup[1]]
  if len(common_elements) >= threshold_filepath:
    edges_list.append((tup[0], tup[1]))
    edges_list.append((tup[1], tup[0]))
    nodes_dict[tup[0]] = True
    nodes_dict[tup[1]] = True

sc_sql = SQLContext(sc)
nodes_list = [(node,) for node in nodes_dict.keys()]
v_df = sc_sql.createDataFrame(nodes_list, ['id'])
e_df = sc_sql.createDataFrame(edges_list, ['src', 'dst'])
graph_frame = GraphFrame(v_df, e_df)
result = graph_frame.labelPropagation(maxIter=5)

communities_df = (result.withColumn('community', result.label.cast('string')).groupBy('community').agg(collect_list('id').alias('members')))
communities_df = communities_df.withColumn('members', sort_array('members'))
communities_df = communities_df.withColumn('size', size('members'))
sorted_communities_df = (communities_df.orderBy(asc('size'), asc('members')))
sorted_communities = (sorted_communities_df.select('members').rdd.flatMap(lambda x: x).collect())

with open(output_filepath, 'w') as textfile:
    for community in sorted_communities:
        community_line = ', '.join(f"'{user_id}'" for user_id in community)
        textfile.write(community_line + '\n')
e_t = time.time()
print('Duration: ', e_t - s_t)