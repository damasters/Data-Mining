import shutil
import itertools
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
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

input_filename = sys.argv[1]
input_clusters = int(sys.argv[2])
output_filename = sys.argv[3]

def loading_data_rand(input_file): 
  with open(input_filename, "r") as file:
    lines = np.array(file.readlines())
  parsed_d = []
  for line in lines:
    parsed_l = np.array(line.strip('\n').split(','))
    parsed_d.append(parsed_l)
  return parsed_d

start_time = time.time()
loaded_data = loading_data_rand(input_filename)
parsed_data = np.array(loaded_data).astype(np.float64)

def creating_splits(parsed_d):
  np.random.shuffle(parsed_d)
  divided_data = np.array_split(parsed_d, 5)
  return divided_data

divided_data = creating_splits(parsed_data)
first_slice = divided_data[0]

dense_cluster_summary, dense_cluster_center, dense_cluster_deviation, dense_cluster_points = {}, {}, {}, {}
sparse_cluster_summary, sparse_cluster_center, sparse_cluster_deviation, sparse_cluster_points = {}, {}, {}, {}
cluster_groups = defaultdict(list)
threshold_distance = 0
removed_set = set()
five_times_input_clust = 5 * input_clusters
initial_kmeans = KMeans(n_clusters=five_times_input_clust).fit(first_slice[:, 2:])
for index, cluster_id in enumerate(initial_kmeans.labels_):
    cluster_groups[cluster_id].append(index)
for indices in cluster_groups.values():
    if len(indices) == 1:
        removed_set.add(indices[0])
cluster_groups = defaultdict(list)
dense_samples = np.delete(first_slice, list(removed_set), axis=0)
second_kmeans = KMeans(n_clusters=input_clusters).fit(dense_samples[:, 2:])
for index, cluster_id in enumerate(second_kmeans.labels_):
    cluster_groups[cluster_id].append(index)

for cluster_id, indices in cluster_groups.items():
    count = len(indices)
    features = dense_samples[indices, 2:]
    total = np.sum(features, axis=0)
    total_sq = np.sum(np.square(features), axis=0)
    dense_cluster_summary[cluster_id] = [count, total, total_sq]
    points = np.array(dense_samples[indices, 0]).astype(int).tolist()
    dense_cluster_points[cluster_id] = points
    center = total / count
    deviation = np.sqrt(np.subtract(total_sq / count, np.square(center)))
    dense_cluster_center[cluster_id] = center
    dense_cluster_deviation[cluster_id] = deviation


remaining_samples = first_slice[list(removed_set), :]
if len(removed_set) >= five_times_input_clust:
    temp_kmeans = KMeans(n_clusters=five_times_input_clust).fit(remaining_samples[:, 2:])
    cluster_groups = {}
    for index, cluster_id in enumerate(temp_kmeans.labels_):
        cluster_groups.setdefault(cluster_id, []).append(index)
    removed_set = {idx for indices in cluster_groups.values() if len(indices) == 1 for idx in indices}
    for cluster_id, indices in cluster_groups.items():
        if len(indices) > 1:
            features = remaining_samples[indices, 2:]
            count = len(indices)
            total = np.sum(features, axis=0)
            total_sq = np.sum(np.square(features), axis=0)

            sparse_cluster_summary[cluster_id] = [count, total, total_sq]
            sparse_cluster_points[cluster_id] = np.array(remaining_samples[indices, 0]).astype(int).tolist()
            sparse_cluster_center[cluster_id] = total / count
            sparse_cluster_deviation[cluster_id] = np.sqrt(total_sq / count - np.square(sparse_cluster_center[cluster_id]))
            
with open(output_filename, "w") as f:
    f.write('The intermediate results:\n')
    num_dense_samples = sum([v[0] for v in dense_cluster_summary.values()])
    num_sparse_clusters = len(sparse_cluster_summary)
    num_sparse_samples = sum([v[0] for v in sparse_cluster_summary.values()])
    num_removed_samples = len(removed_set)
    round_1_results = f'Round 1: {num_dense_samples},{num_sparse_clusters},{num_sparse_samples},{num_removed_samples}\n'
    f.write(round_1_results)

distance_threshold = 2 * np.sqrt(first_slice.shape[1] - 2)

def update_cluster(cluster_summary, cluster_center, cluster_deviation, cluster_points, cluster_id, data, index):
    if cluster_id not in cluster_summary:
        cluster_summary[cluster_id] = [0, np.zeros_like(data), np.zeros_like(data)]
        cluster_center[cluster_id] = np.zeros_like(data)
        cluster_deviation[cluster_id] = np.ones_like(data)
        cluster_points[cluster_id] = []

    count, total, total_sq = cluster_summary[cluster_id]
    new_total = total + data
    new_total_sq = total_sq + np.square(data)

    cluster_summary[cluster_id] = [count + 1, new_total, new_total_sq]
    updated_center = new_total / (count + 1)
    updated_deviation = np.sqrt(new_total_sq / (count + 1) - np.square(updated_center))

    cluster_center[cluster_id] = updated_center
    cluster_deviation[cluster_id] = updated_deviation
    cluster_points[cluster_id].append(index)

def compute_mahalanobis_distance(data, center, deviation):
    return np.sqrt(np.sum(np.square((data - center) / deviation)))

def find_nearest_cluster(data, cluster_summary, cluster_center, cluster_deviation):
    min_distance = float('inf')
    selected_cluster = -1
    for cluster_id, _ in cluster_summary.items():
        distance = compute_mahalanobis_distance(data, cluster_center[cluster_id], cluster_deviation[cluster_id])
        if distance < min_distance:
            min_distance = distance
            selected_cluster = cluster_id
    return selected_cluster, min_distance

def process_removed_samples(divided_data_round, removed_set, threshold, cluster_summary, cluster_center, cluster_deviation, cluster_points):
    current_removed_samples = divided_data_round[list(removed_set), :]

    if len(removed_set) >= threshold:
        temp_kmeans = KMeans(n_clusters=threshold).fit(current_removed_samples[:, 2:])
        cluster_groups = defaultdict(list)
        for index, cluster_id in enumerate(temp_kmeans.labels_):
            cluster_groups[cluster_id].append(index)

        removed_set.clear()
        for indices in cluster_groups.values():
            if len(indices) == 1:
                original_index = np.where(divided_data_round == current_removed_samples[indices[0]])[0][0]
                removed_set.add(original_index)
            else:
                for idx in indices:
                    update_cluster(cluster_summary, cluster_center, cluster_deviation, cluster_points, cluster_id, current_removed_samples[idx, 2:], current_removed_samples[idx, 0])

def merge_clusters(dense_summary, dense_center, dense_deviation, dense_points, 
                   sparse_summary, sparse_center, sparse_deviation, sparse_points, threshold):
    merge_dict = {}
    for cluster_id_one in sparse_summary.keys():
        closest_cluster = -1
        closest_distance = threshold
        for cluster_id_two in dense_summary.keys():
            if cluster_id_one != cluster_id_two:
                dist1 = compute_mahalanobis_distance(sparse_center[cluster_id_one], dense_center[cluster_id_two], dense_deviation[cluster_id_two])
                dist2 = compute_mahalanobis_distance(dense_center[cluster_id_two], sparse_center[cluster_id_one], sparse_deviation[cluster_id_one])
                mahalanobis_dist = min(dist1, dist2)
                if mahalanobis_dist < closest_distance:
                    closest_distance = mahalanobis_dist
                    closest_cluster = cluster_id_two
        merge_dict[cluster_id_one] = closest_cluster

    for cluster_id_one, cluster_id_two in merge_dict.items():
        if cluster_id_one in sparse_summary and cluster_id_two in dense_summary:
            if cluster_id_one != cluster_id_two and merge_dict[cluster_id_one] != -1:
                count = sparse_summary[cluster_id_one][0] + dense_summary[cluster_id_two][0]
                total = np.add(sparse_summary[cluster_id_one][1], dense_summary[cluster_id_two][1])
                total_sq = np.add(sparse_summary[cluster_id_one][2], dense_summary[cluster_id_two][2])

                dense_summary[cluster_id_two] = [count, total, total_sq]

                center = total / count
                deviation = np.sqrt(np.subtract(total_sq / count, np.square(center)))

                dense_center[cluster_id_two] = center
                dense_deviation[cluster_id_two] = deviation
                dense_points[cluster_id_two].extend(sparse_points[cluster_id_one])

                del sparse_summary[cluster_id_one]
                del sparse_center[cluster_id_one]
                del sparse_deviation[cluster_id_one]
                del sparse_points[cluster_id_one]

def append_results_to_file(filename, round_number, dense_summary, sparse_summary, removed_set):
    with open(filename, "a") as f:
        num_dense_samples = sum([v[0] for v in dense_summary.values()])
        num_sparse_clusters = len(sparse_summary)
        num_sparse_samples = sum([v[0] for v in sparse_summary.values()])
        num_removed_samples = len(removed_set)
        result_str = f'Round {round_number}: {num_dense_samples},{num_sparse_clusters},{num_sparse_samples},{num_removed_samples}\n'
        f.write(result_str)
  
for round_number in range(2, 6):
    for index, value in enumerate(divided_data[round_number - 1]):
        current_data = value[2:]
        nearest_cluster, min_distance = find_nearest_cluster(current_data, dense_cluster_summary, dense_cluster_center, dense_cluster_deviation)

        if min_distance < distance_threshold and nearest_cluster != -1:
            update_cluster(dense_cluster_summary, dense_cluster_center, dense_cluster_deviation, dense_cluster_points, nearest_cluster, current_data, int(value[0]))
        else:
            nearest_cluster, min_distance = find_nearest_cluster(current_data, sparse_cluster_summary, sparse_cluster_center, sparse_cluster_deviation)
            if min_distance < distance_threshold and nearest_cluster != -1:
                update_cluster(sparse_cluster_summary, sparse_cluster_center, sparse_cluster_deviation, sparse_cluster_points, nearest_cluster, current_data, int(value[0]))
            else:
                removed_set.add(index)
    process_removed_samples(divided_data[round_number - 1], removed_set, five_times_input_clust, sparse_cluster_summary, sparse_cluster_center, sparse_cluster_deviation, sparse_cluster_points)
    merge_clusters(dense_cluster_summary, dense_cluster_center, dense_cluster_deviation, dense_cluster_points, sparse_cluster_summary, sparse_cluster_center, sparse_cluster_deviation, sparse_cluster_points, distance_threshold)
    append_results_to_file(output_filename, round_number, dense_cluster_summary, sparse_cluster_summary, removed_set)

final_results = {}
for cluster_id in dense_cluster_summary.keys():
    for point in dense_cluster_points[cluster_id]:
        final_results[point] = cluster_id
for cluster_id in sparse_cluster_summary.keys():
    for point in sparse_cluster_points[cluster_id]:
        final_results[point] = -1
for point in removed_set:
    final_results[point] = -1

with open(output_filename, "a") as f:
    f.write('\nThe clustering results:\n')
    for point in sorted(final_results, key=int):
        cluster_id = final_results[point]
        f.write(f'{point},{cluster_id}\n')           

end_time = time.time()
print('Duration: ', end_time - start_time)