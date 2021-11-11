from pyspark import SparkContext
import sys
import time
from itertools import combinations
from collections import defaultdict

'''
    check if user_index_1 and user_index_2 have more than filter_threshold common businesses
'''
def hasEdge(user_index_1, user_index_2, user_business_dict, filter_threshold):
    return len(set(user_business_dict[user_index_1]).intersection(set(user_business_dict[user_index_2]))) >= filter_threshold

'''
    give a root, run dfs on graph
    level_order: bfs spanning tree from root
    vertex_parent_dict: for each vertex, its parent vertices set
    vertex_path_num_dict: num of shortest paths at vertex
    returns (level_order, vertex_parent_dict, vertex_path_num_dict)
'''
def bfs(graph_dict, root):
    # data structures for bfs/level order
    visited = set()
    queue = list()
    level_num = 1

    # level order results
    # [[level_1_vertices], [level_2_vertices]]
    levels = list()
    # {vertex: level_num}
    vertex_level_dict = dict()
    # {vertex: num_shortest_paths}
    vertex_path_num_dict = dict()
    # {vertex: {parent_vertex}}
    vertex_parent_dict = defaultdict(set)

    visited.add(root)
    queue.append(root)
    vertex_path_num_dict[root] = 1

    while queue:
        size = len(queue)
        level_num += 1
        level = list()
        for i in range(size):
            curr = queue.pop(0)
            vertex_level_dict[curr] = level_num
            level.append(curr)
            for neighbor in graph_dict[curr]:
                # mark curr as parent (for any neighbor)
                if vertex_level_dict.get(neighbor) == level_num - 1:
                    vertex_parent_dict[curr].add(neighbor)
                # if neighbor node not visited, neightbor as child
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
            # sum up shortest paths of curr's parent
            if curr != root:
                path_count = 0
                for parent_vertex in vertex_parent_dict[curr]:
                    path_count += vertex_path_num_dict[parent_vertex]
                vertex_path_num_dict[curr] = path_count
        levels.append(level)

    return (root, (levels, vertex_parent_dict, vertex_path_num_dict))

'''
    for each node, calculate edge betweenness from bottom-up
'''
def calculate_edge_betweenness(vertices, levels, vertex_parent_dict, vertex_path_num_dict):
    vertex_credit_dict = { vertex: 1 for vertex in vertices }
    level_num = len(levels)

    # [("vertex_1,vertex_2", betweenness)]
    edge_betweenness = list()

    # start from bottom level to top level - 1
    for i in range(level_num):
        for curr in levels[level_num - i - 1]:
            for parent in vertex_parent_dict[curr]:
                betweenness = vertex_credit_dict[curr] * (vertex_path_num_dict[parent] / vertex_path_num_dict[curr])
                vertex_credit_dict[parent] += betweenness
                # mark A -> B as B -> A same, count twice
                key = sorted([curr, parent])
                edge_betweenness.append((str(key[0]) + "," + str(key[1]), betweenness))

    return edge_betweenness

sc = SparkContext('local[*]', 'task2')
sc.setLogLevel("OFF")

start_time = time.time()

filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
betweenness_output_file_path = sys.argv[3]
community_output_file_path = sys.argv[4]

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
    .sortBy(lambda user_id: user_id) \
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
    .sortBy(lambda business_id: business_id) \
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

'''
    Generate vertices: users in edges
'''
vertices_RDD = edges_RDD \
    .flatMap(lambda pair: list(pair)) \
    .distinct() \
    .map(lambda vertex: vertex) \
    .cache()
vertices = vertices_RDD.collect()

'''
    Generate graph_dict: { vertex: [neigtbor_vertex] }
    Represent by adjacent list
'''
graph_RDD = edges_RDD \
    .flatMap(lambda pair: [(pair[0], pair[1]), (pair[1], pair[0])]) \
    .distinct() \
    .groupByKey() \
    .map(lambda item: (item[0], set(item[1]))) \
    .cache()
graph_dict = graph_RDD.collectAsMap()

'''
    4.3.1 Betweenness Calculation
'''
betweenness_RDD = vertices_RDD \
    .map(lambda root: bfs(
        graph_dict = graph_dict, 
        root = root
    )) \
    .flatMap(lambda item: calculate_edge_betweenness(
        vertices = vertices, 
        levels = item[1][0], 
        vertex_parent_dict = item[1][1],
        vertex_path_num_dict = item[1][2]
    )) \
    .reduceByKey(lambda accu, curr: accu + curr) \
    .map(lambda edge_betweenness: (edge_betweenness[0], edge_betweenness[1] / 2)) \
    .sortBy(lambda item: -item[1]) \
    .cache()

betweenness_result = betweenness_RDD \
    .map(lambda item: [
            index_user_dict[int(item[0].split(',')[0])], 
            index_user_dict[int(item[0].split(',')[1])],
            round(item[1], 5)
        ]) \
    .sortBy(lambda item: (-item[2], item[0])) \
    .collect()

with open(betweenness_output_file_path, 'w') as betweenness_output_file:
    for result in betweenness_result:
        betweenness_output_file.write('(\''+ result[0] +'\', \'' + result[1]+ '\'), ' + str(result[2]) +'\n')
betweenness_output_file.close()    

'''
    4.3.2 Community Detection
'''
# m: number of edges in the original graph
m = edges_RDD.count()
num_betweenness = m
# A: adjacent matrix of the original graph
A = graph_RDD.collectAsMap()

max_modularity = -1
communities_result = list()

while True: 
    '''
        Start with the graph and all its edges and remove edges with highest betweenness
    '''
    # [("vertex_1,vertex_2", betweenness)]
    edge_betweenness = betweenness_RDD.collect()
    max_betweenness = betweenness_RDD.first()[1]

    '''
        Removing the edges with the highest betweenness
    '''
    for item in edge_betweenness:
        if item[1] == max_betweenness:
            vertex_1 = int(item[0].split(',')[0])
            vertex_2 = int(item[0].split(',')[1])
            # remove edge in non-directed graph
            graph_dict[vertex_1].remove(vertex_2)
            graph_dict[vertex_2].remove(vertex_1)
            num_betweenness -= 1

    if num_betweenness == 0:
        break
    else:
        '''
            Detect communities using bfs
        '''
        current_communities = list()
        visited = set()
        for vertex in vertices:
            if vertex in visited:
                continue
            community = bfs(root = vertex, graph_dict = graph_dict)[1][0]
            community_set = set()
            for level in community:
                for vertex in level:
                    community_set.add(vertex)
            current_communities.append(community_set)
            # If a vertex is detected once, mark as visited
            visited = visited.union(community_set)
        
        '''
            Compute modularity
            Q(G, S) = (1/2m) * 
                sum([community in communities]
                        [vertex_i in community]
                            [vertex j in community]
                                (a_ij - (degree_of(vertex_i) * degree_of(vertex_j) / 2m)))
        '''
        modularity = 0
        for community in current_communities:
            for vertex_i in community:
                for vertex_j in community:
                    a_ij = 0
                    # if there's an edge betwwen vertex_i and vertex_j
                    if vertex_j in A[vertex_i] or vertex_i in A[vertex_j]:
                        a_ij = 1 
                    modularity += (a_ij - (len(A[vertex_i]) * len(A[vertex_j]) / (2 * m)))
        modularity = (1 / (2 * m)) * modularity
        
        '''
            Update community detection with max modularity
        '''
        if modularity > max_modularity:
            max_modularity = modularity
            communities_result = current_communities

        '''
            After removing the edge with max betweenness, re-compute betweenness in the graph
        '''
        betweenness_RDD = vertices_RDD \
            .map(lambda root: bfs(
                graph_dict = graph_dict, 
                root = root
            )) \
            .flatMap(lambda item: calculate_edge_betweenness(
                vertices = vertices, 
                levels = item[1][0], 
                vertex_parent_dict = item[1][1],
                vertex_path_num_dict = item[1][2]
            )) \
            .reduceByKey(lambda accu, curr: accu + curr) \
            .map(lambda edge_betweenness: (edge_betweenness[0], edge_betweenness[1] / 2)) \
            .sortBy(lambda item: -item[1]) 
        
sorted_communities_result = list()
for community in communities_result:
    sorted_communities_result.append(sorted(list(community)))
sorted_communities_result = sorted(sorted_communities_result, key = lambda community: (len(community), community[0]))

with open(community_output_file_path, 'w') as community_output_file:
    for result in sorted_communities_result:
        size = len(result)
        for i in range(size - 1):
            community_output_file.write('\'' + index_user_dict[result[i]] + '\', ')
        community_output_file.write('\'' + index_user_dict[result[size - 1]] + '\'\n')
    community_output_file.close()

print("Duration:", str(time.time() - start_time))