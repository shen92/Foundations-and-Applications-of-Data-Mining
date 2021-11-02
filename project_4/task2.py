from pyspark import SparkContext
from pyspark.sql import SQLContext
import sys
import time
from itertools import combinations
from graphframes import *

'''
    check if user_index_1 and user_index_2 have more than filter_threshold common businesses
'''
def hasEdge(user_index_1, user_index_2, user_business_dict, filter_threshold):
    return len(set(user_business_dict[user_index_1]).intersection(set(user_business_dict[user_index_2]))) >= filter_threshold

sc = SparkContext('local[*]', 'task1')
sc.setLogLevel("OFF")

sqlc = SQLContext(sc)

start_time = time.time()

filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
community_output_file_path = sys.argv[3]

# 0:user_id, 1:business_id
data_RDD = sc.textFile(input_file_path)
header = data_RDD.first()
data_RDD = data_RDD.filter(lambda item: item != header).cache()

'''
    Create user_index_dict and index_user_dict
    1) map: user_id
    2) distinct: user_id
    3) zipWithIndex: user_id: user_index 
'''
# user_index_dict: {user_id: index}
user_index_dict = data_RDD \
    .map(lambda row: row.split(',')[0]) \
    .distinct() \
    .zipWithIndex() \
    .collectAsMap()
# index_user_dict: {index: user_id}
index_user_dict = { index: user_id for user_id, index in user_index_dict.items() }

'''
    Create business_index_dict and index_business_dict
    1) map: business_id
    2) distinct: business_id
    3) zipWithIndex: business_id: business_index 
'''
# business_index_dict: {business_id: index}
business_index_dict = data_RDD \
    .map(lambda row: row.split(',')[1]) \
    .distinct() \
    .zipWithIndex() \
    .collectAsMap()
# index_business_dict: {index: business_id}
# index_business_dict = { index: business_id for business_id, index in user_index_dict.items() }

'''
    Create {user_index:[business_index]} dict
    1) map: (user_index, business_index)
    2) groupByKey: (user_index: [business_index])
'''
user_business_dict = data_RDD \
    .map(lambda row: (user_index_dict[row.split(',')[0]], business_index_dict[row.split(',')[1]])) \
    .groupByKey() \
    .collectAsMap()

'''
    Create user_index_pairs_RDD: (user_index_1, user_index_2)
    -> all combinations of any 2 user_index
'''
user_index_pair_RDD = sc.parallelize(list(combinations(user_index_dict.values(), 2))) \
    .cache()

'''
    Generate edges: for a (user_id_1, user_id_2), intersection of common business is greater than filter_threshold
'''
edges_RDD = user_index_pair_RDD \
    .filter(lambda pair: hasEdge(pair[0], pair[1], user_business_dict, filter_threshold)) \
    .cache()
edges = edges_RDD \
    .collect()
edges_df = sqlc.createDataFrame(edges, ['src', 'dst'])

'''
    Generate vertices: users in edges
'''
vertices = edges_RDD \
    .flatMap(lambda pair: list(pair)) \
    .distinct() \
    .map(lambda vertex: [vertex]) \
    .collect()
vertices_df = sqlc.createDataFrame(vertices, ['id'])

'''
    Label Propagation Algorithm
'''
graph = GraphFrame(vertices_df, edges_df)
# (user_index, community_id)
communities = graph.labelPropagation(maxIter=5)

results = communities.rdd \
    .map(lambda result: (result[1], index_user_dict[result[0]])) \
    .groupByKey() \
    .map(lambda user_community: sorted(user_community[1])) \
    .sortBy(lambda user_ids: (len(user_ids), user_ids[0])) \
    .collect()

with open(community_output_file_path, 'w') as community_output_file:
    for result in results:
        size = len(result)
        for i in range(size - 1):
            community_output_file.write('\'' + result[i] + '\', ')
        community_output_file.write('\'' + result[size - 1] + '\'\n')
    community_output_file.close()

print("Duration:", str(time.time() - start_time))