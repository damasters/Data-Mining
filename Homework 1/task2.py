# -*- coding: utf-8 -*-

from pyspark import SparkContext, SparkConf
import json
import sys
import time

if 'sc' in globals():
    sc.stop()
spark_conf = SparkConf().setAppName('TestReviewSet').setMaster('local')
sc = SparkContext(conf=spark_conf)
sc.setLogLevel("WARN")
review_filepath = sys.argv[1]
output_filepath = sys.argv[2]
n_partitions = sys.argv[3]
test_review_data = sc.textFile(review_filepath)

review_data_dict = test_review_data.map(lambda x: json.loads(x))

num_partitions_def = review_data_dict.getNumPartitions()
items_per_part_def = review_data_dict.glom().map(len).collect()

start_time_def = time.time()
top_businesses_default = review_data_dict \
    .map(lambda x: (x.get('business_id'), 1)).filter(lambda x: x[0] is not None and x[0].strip() != "") \
    .reduceByKey(lambda x,y: x+y).map(lambda x: (-x[1], x[0])) \
    .sortBy(lambda x: x).map(lambda x: (x[1], -x[0])).take(10)

end_time_def = time.time()
default_time = end_time_def - start_time_def

map_by_key = review_data_dict.map(lambda x: (x.get('business_id'), x)).filter(lambda x: x[0] is not None and x[0].strip() != "")

try:
    num_partitions_cus = int(round(float(n_partitions)))
    if num_partitions_cus <= 0: #partitions = zero throw error
        num_partitions_cus = 1  #goes to 1 if partition is less than or equal to 0
except ValueError:
    num_partitions_cus = 1  #if error then defaults to 1
    
custom_rdd = map_by_key.partitionBy(num_partitions_cus, lambda x: hash(x)%num_partitions_cus)
items_per_part_cus = custom_rdd.glom().map(len).collect()
start_time_cus = time.time()

top_businesses_custom = custom_rdd \
    .map(lambda x: (x[0], 1)) \
    .reduceByKey(lambda x,y: x+y).map(lambda x: (-x[1], x[0])) \
    .sortBy(lambda x: x).map(lambda x: (x[1], -x[0])).take(10)

end_time_cus = time.time()
custom_time = end_time_cus - start_time_cus

task_two_output = {
    "default": {
        "n_partition": num_partitions_def,
        "n_items": list(items_per_part_def),
        "exe_time": default_time
    },
    "customized": {
        "n_partition": num_partitions_cus,
        "n_items": list(items_per_part_cus),
        "exe_time": custom_time
    }
}
task_two_json_format = json.dumps(task_two_output, indent=4)

with open(output_filepath, 'w') as outfile:
    outfile.write(task_two_json_format)