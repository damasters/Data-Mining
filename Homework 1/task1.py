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
test_review_data = sc.textFile(review_filepath)

key_list = ['n_review', 'n_review_2018', 'n_user', 'top10_user', 'n_business', 'top10_business']
answers_dict = {key:None for key in key_list}

#make each line of the json a k,v pair in a dictionary
#now review_data_dict is an RDD which contains dictionaries of each row in the json
review_data_dict = test_review_data.map(lambda x: json.loads(x))

#Question A: The total number of reviews. OUTPUT: a number
tot_num_reviews = review_data_dict.count()
answers_dict["n_review"] = tot_num_reviews

#Question B: The number or reviews in 2018. OUTPUT: a number
num_reviews_date = review_data_dict.map(lambda x: x.get('date'))\
    .filter(lambda x: x is not None and x.strip() != "" and x.startswith('2018')).count()
answers_dict["n_review_2018"] = num_reviews_date

# Question C: The number of distinct users who wrote reviews:
count_dist_usrs = review_data_dict.map(lambda x: x.get('user_id')) \
    .filter(lambda x: x is not None and x.strip() != "").distinct().count()
answers_dict["n_user"] = count_dist_usrs

# Question D: The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
top_ten_list = review_data_dict \
    .map(lambda x: (x.get('user_id'), 1)).filter(lambda x: x[0] is not None and x[0].strip() != "")\
    .reduceByKey(lambda x,y: x+y) \
    .map(lambda x: (-x[1], x[0])).sortBy(lambda x: x) \
    .map(lambda x: (x[1], -x[0])).take(10)
answers_dict["top10_user"] = top_ten_list

# Question E: The number of distinct businesses that have been reviewed
distinct_businesses = review_data_dict.map(lambda x: x.get('business_id'))\
    .filter(lambda x: x is not None and x.strip() != "").distinct().count()
answers_dict["n_business"] = distinct_businesses

# Question F: The top 10 businesses that had the largest numbers of reviews and the number of reviews they had
top_ten_bus_list = review_data_dict \
    .map(lambda x: (x.get('business_id'), 1)).filter(lambda x: x[0] is not None and x[0].strip() != "") \
    .reduceByKey(lambda x,y: x+y) \
    .map(lambda x: (-x[1], x[0])) \
    .sortBy(lambda x: x) \
    .map(lambda x: (x[1], -x[0])) \
    .take(10)
answers_dict["top10_business"] = top_ten_bus_list

answers_json = json.dumps(answers_dict, indent=4)

with open(output_filepath, 'w') as outfile:
    outfile.write(answers_json)
