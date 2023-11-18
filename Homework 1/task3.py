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
business_filepath = sys.argv[2]
output_a_filepath = sys.argv[3]
output_b_filepath = sys.argv[4]
test_review_data = sc.textFile(review_filepath)

review_data_dict = test_review_data.map(lambda x: json.loads(x))

#Part A)
business_data = sc.textFile(business_filepath)
business_data_dict = business_data.map(lambda x: json.loads(x))

review_data_stars = review_data_dict.map(lambda x: (x['business_id'], x['stars']))
business_data_city = business_data_dict.map(lambda x: (x['business_id'], x['city']))

joined = review_data_stars.join(business_data_city) \
    .map(lambda x:  (x[1][1], (x[1][0], 1))) \
    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
    .mapValues(lambda x: x[0]/x[1]).sortBy(lambda x: (-x[1], x[0])) \
    .map(lambda x: x[0]+','+str(x[1])).collect()


with open(output_a_filepath, 'w') as outfile:
  outfile.write("city,stars")
  for a in joined:
    outfile.write('\n'+a)

#Part B)
key_list_stars = ['m1', 'm2', 'reason']
answers_dict_stars = {key:None for key in key_list_stars}

#M1
start_time_m1 = time.time()
test_review_data = sc.textFile(review_filepath)
business_data = sc.textFile(business_filepath)
business_data_dict = business_data.map(lambda x: json.loads(x))
review_data_dict = test_review_data.map(lambda x: json.loads(x))
business_data_city = business_data_dict.map(lambda x: (x.get('business_id'), x.get('city'))).filter(lambda x: x[0] is not None and x[0].strip() != '' and x[1] is not None)
review_data_stars = review_data_dict.map(lambda x: (x.get('business_id'), x.get('stars'))).filter(lambda x: x[0] is not None and x[0].strip() != '' and x[1] is not None)

avg_m1 = review_data_stars.join(business_data_city) \
    .map(lambda x:  (x[1][1], (x[1][0], 1))) \
    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
    .mapValues(lambda x: x[0]/x[1]).collect()

top_cities_m1_sort = sorted(avg_m1, key=lambda x: (-x[1], x[0]))[:10]
print(top_cities_m1_sort)
end_time_m1 = time.time()
execution_m1 = end_time_m1 - start_time_m1

#M2
start_time_m2 = time.time()
test_review_data = sc.textFile(review_filepath)
business_data = sc.textFile(business_filepath)
business_data_dict = business_data.map(lambda x: json.loads(x))
review_data_dict = test_review_data.map(lambda x: json.loads(x))
business_data_city = business_data_dict.map(lambda x: (x['business_id'], x['city']))
review_data_stars = review_data_dict.map(lambda x: (x['business_id'], x['stars']))
avg_m2 = review_data_stars.join(business_data_city) \
    .map(lambda x:  (x[1][1], (x[1][0], 1))) \
    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
    .mapValues(lambda x: x[0]/x[1]).sortBy(lambda x: (-x[1], x[0])).take(10)
print(avg_m2)
end_time_m2 = time.time()
execution_m2 = end_time_m2 - start_time_m2
reason = "The observed execution times show the difference between native Python sorting and Spark's distributed sorting. While Python is quickier than Spark for smaller datasets, Spark can handle huge datasets quickier than standard Python due to RDD's ability to process data with parallel partitions."
task_b_output = {"m1": execution_m1, "m2": execution_m2, "reason": reason}
task_b_out_pretty = json.dumps(task_b_output, indent=4)
with open(output_b_filepath, "w") as file:
    file.write(task_b_out_pretty)
sc.stop()