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

if 'sc' in globals():
    sc.stop()
spark_conf = SparkConf().setAppName('Assignment4').setMaster('local')
sc = SparkContext(conf=spark_conf)
sc.setLogLevel('ERROR')

threshold_filepath = int(sys.argv[1])
input_filepath = sys.argv[2]
betweenness_output_file_path = sys.argv[3]
output_filepath = sys.argv[4]

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

graph = defaultdict(set)
for edge in edges_list:
    graph[edge[0]].add(edge[1])
    graph[edge[1]].add(edge[0])

def bfs_shortest_paths(graph, start_node):
    depth = {start_node: 0}
    path_count = defaultdict(int, {start_node: 1})
    ancestors = defaultdict(list)
    search = [start_node]
    history = [start_node]

    while search:
        node = search.pop(0)
        for adj_node in graph[node]:
            if adj_node not in depth:
                search.append(adj_node)
                depth[adj_node] = depth[node] + 1
                history.append(adj_node)
            if depth[adj_node] == depth[node] + 1:
                path_count[adj_node] += path_count[node]
                ancestors[adj_node].append(node)
    return history, ancestors, path_count

def calculate_edge_cont(trav_history, ancestors, path_count):
    weights = defaultdict(int, {n: 1 for n in trav_history})
    edge_cont = defaultdict(int)
    for node in reversed(trav_history):
        for pred in ancestors[node]:
            flow_val = (weights[node] * path_count[pred]) / path_count[node]
            weights[pred] += flow_val
            sorted_edge = tuple(sorted([node, pred]))
            edge_cont[sorted_edge] += flow_val
    return edge_cont

def update_betweenness(betweenness, edge_cont):
    for edge, value in edge_cont.items():
        betweenness[edge] += value / 2
    return betweenness

def compute_betweenness(graph, nodes):
    betweenness = defaultdict(float)
    for start_node in nodes:
        trav_history, ancestors, path_count = bfs_shortest_paths(graph, start_node)
        edge_cont = calculate_edge_cont(trav_history, ancestors, path_count)
        betweenness = update_betweenness(betweenness, edge_cont)
    return sorted(betweenness.items(), key=lambda x: (-x[1], sorted(x[0])))

edge_btwness = compute_betweenness(graph, nodes_dict.keys())
edge_btwness_round = [(edge, round(score, 5)) for edge, score in edge_btwness]

with open(betweenness_output_file_path, 'w') as output:
  for tup in edge_btwness_round:
    output.write(str(tup[0])+','+str(tup[1])+'\n')

def find_communities(network):
    communities_list = []
    nodes_to_visit = set(network.keys())
    while nodes_to_visit:
        connected_nodes = set()
        exploring_nodes = {nodes_to_visit.pop()}
        while exploring_nodes:
            node = exploring_nodes.pop()
            connected_nodes.add(node)
            neighbours = network[node]
            exploring_nodes.update(neighbours - connected_nodes)
            nodes_to_visit -= neighbours
        communities_list.append(sorted(connected_nodes))
    return communities_list

def modularity_calculation(communities, initial_network, total_edges):
    modularity_value = 0.0
    for community in communities:
        for node1 in community:
            for node2 in community:
                if node2 in initial_network[node1]:
                    a_ij = 1
                else:
                    a_ij = 0
                modularity_value += a_ij - (len(initial_network[node1]) * len(initial_network[node2])) / (2 * total_edges)
    return modularity_value / (2 * total_edges)

def girvan_newman_main(network, initial_node_set):
    network_copy = copy.deepcopy(network)
    total_edges = sum(len(neighbours) for neighbours in network.values()) / 2
    highest_modularity = -1
    optimal_communities = []
    betweenness = compute_betweenness(network_copy, initial_node_set)
    while betweenness:
        max_betweenness = max(betweenness, key=lambda x: x[1])[1]
        for edge, value in betweenness:
            if value == max_betweenness:
                network_copy[edge[0]].remove(edge[1])
                network_copy[edge[1]].remove(edge[0])
        communities = find_communities(network_copy)
        modularity = modularity_calculation(communities, network, total_edges)
        if modularity > highest_modularity:
            highest_modularity = modularity
            optimal_communities = communities
        betweenness = compute_betweenness(network_copy, set(network_copy.keys()))
    return optimal_communities

detected_communities = girvan_newman_main(graph, set(nodes_dict.keys()))

with open(output_filepath, 'w') as output_file:
    for community in sorted(detected_communities, key=lambda x: (len(x), x)):
        comm_formatted = ', '.join(f"'{mem}'" for mem in community)
        output_file.write(comm_formatted + '\n')
e_t = time.time()
print('Duration: ', e_t - s_t)