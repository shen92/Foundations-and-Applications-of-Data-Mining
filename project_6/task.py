from pyspark import SparkContext
from sklearn.cluster import KMeans
import sys
import time
import math

'''
    For a list of string of features, convert it to float
'''
def format_features(features):
    formatted_features = list()
    for feature in features:
        formatted_features.append(float(feature))
    return formatted_features

'''
    Used for data load and RS -> no label item included
'''
def get_features_from_line(data):
    features = list()
    for line in data:
        features.append(line[1])
    return features

'''
    For a point's features, calculate its mahalanobis_distance to each DS/CS
    sigma: (SUMSQi / N) – (SUMi / N) ^ 2
'''
def compute_mahalanobis_distance_between_point_and_cluster(features, N_dict, SUM_dict, SUMSQ_dict):
    # {label: mahalanobis_distance}
    mahalanobis_distance_dict = dict()

    for label in N_dict.keys():
        N = N_dict[label]
        SUM = SUM_dict[label]
        SUMSQ = SUMSQ_dict[label]

        sum = 0
        for i in range(len(features)):
            centroid = SUM[i] / N
            sigma = (SUMSQ[i] / N - (SUM[i] / N) ** 2) ** 0.5
            sum += ((features[i] - centroid) / sigma) ** 2
        
        mahalanobis_distance_dict[label] = sum ** 0.5

    return mahalanobis_distance_dict

''' 
    for a data_point_index's mahalanobis_distance_dict: {label: distance}
    return its cloest distance to a DS/CS cluster -> (label, distance)
'''
def get_min_mahalanobis_distance(mahalanobis_distance_dict):
    if mahalanobis_distance_dict:
        sorted_mahalanobis_distance_dict = sorted(mahalanobis_distance_dict.items(), key=lambda x: x[1])
        return sorted_mahalanobis_distance_dict[0]
    return (-1, -1)

def assign_DS_label(mahalanobis_distance, DISTANCE_THRESHOLD):
    if mahalanobis_distance[1] < DISTANCE_THRESHOLD:
        return mahalanobis_distance[0]
    else:
        return -1

def assign_CS_label(mahalanobis_distance, DISTANCE_THRESHOLD):
    if mahalanobis_distance == (-1, -1):
        return "r"
    if mahalanobis_distance[1] < DISTANCE_THRESHOLD:
        return mahalanobis_distance[0]
    else:
        return "r"

sc = SparkContext('local[*]', 'task')
sc.setLogLevel("OFF")

start_time = time.time()

input_file = sys.argv[1]
n_cluster = int(sys.argv[2])
output_file = sys.argv[3]

# num of data loads
NUM_ROUNDS = 5
DISTANCE_THRESHOLD = 2 * (n_cluster ** 0.5)

# (0:data_point_index, 1:[feature])
data_RDD = sc.textFile(input_file) \
    .map(lambda line: line.split(',')) \
    .map(lambda line: (
        int(line[0]),
        format_features(line[2:])
        )) \
    .cache()
data_count = data_RDD.count()
round_size = math.ceil(data_count / NUM_ROUNDS)
round = 0

'''
    Step 1. Load 20% of the data randomly.
    Load the whole data set by 20% sequentially -> Load first 20% data
'''
start_index = round * round_size
end_index = (round + 1) * round_size
round_0_data_RDD = data_RDD \
    .filter(lambda line: start_index <= line[0] and line[0] < end_index) \
    .cache()

'''
    Step 2. Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters) on the data in memory using the Euclidean distance as the similarity measurement.
'''
# round_0_data: (0:data_point_index, 1:[feature])
round_0_data = round_0_data_RDD.collect()
# round_0_data_pre_results.labels_:[label] -> index:data_point_index, value: label
round_0_data_pre_results = KMeans(n_clusters = min(n_cluster * 5, len(round_0_data)), random_state = 1).fit(get_features_from_line(round_0_data))

'''
    Step 3. In the K-Means result from Step 2, move all the clusters that contain only one point to RS (outliers).
'''
# get list of labels of cluster size of 1
# RS_labels: [label]
RS_labels = sc.parallelize(round_0_data_pre_results.labels_) \
    .map(lambda label: (label, 1)) \
    .groupByKey() \
    .filter(lambda label: len(label[1]) == 1) \
    .map(lambda label: label[0]) \
    .collect()

# RS_of_index: [data_point_index]
pre_RS_of_index = list()
# RS: [data_point_index, [feature]]
pre_RS = list()
for i in range(len(round_0_data)):
    # (0:data_point_index, 1:[feature])
    line = round_0_data[i]
    label = round_0_data_pre_results.labels_[i]
    if label in RS_labels:
        pre_RS_of_index.append(round_0_data[i][0])
        pre_RS.append([line[0], line[1]])

'''
    Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
'''
# load_i_without_RS: [(0:data_point_index, 2:[feature])] -> DS of load 0
round_0_data_without_pre_RS = round_0_data_RDD \
    .filter(lambda line: line[0] not in pre_RS_of_index) \
    .collect()
round_0_data_without_pre_RS_results = KMeans(n_clusters = min(n_cluster, len(round_0_data_without_pre_RS)), random_state = 1) \
    .fit(get_features_from_line(round_0_data_without_pre_RS))

# clusters: {data_point_index: label} -> final result
results = dict()
for i in range(len(round_0_data_without_pre_RS)):
    label = round_0_data_without_pre_RS_results.labels_[i]
    data_point_index = round_0_data_without_pre_RS[i][0]
    results[data_point_index] = label


'''
    Step 5. Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and generate statistics).
    
    For each cluster, the discard set (DS) is summarized by:
    N: The number of points
    SUM: the sum of the coordinates of the points
    SUMSQ: the sum of squares of coordinates
'''
# Add label to DS
# DS: [data_point_index, label, [feature]]
DS = list()
for i in range(len(round_0_data_without_pre_RS)):
    line = round_0_data_without_pre_RS[i]
    label = round_0_data_without_pre_RS_results.labels_[i]
    DS.append([line[0], label, line[1]])
DS_RDD = sc.parallelize(DS).cache()
num_DS_points = len(DS)

# DS: [data_point_index, label, [feature]]
# N_DS_dict: {label: num of points}
N_DS_dict = DS_RDD \
    .map(lambda line: (line[1], 1)) \
    .reduceByKey(lambda accu, curr: accu + curr) \
    .collectAsMap()
# N_DS = sorted(N_DS, key = lambda item: item[0])

# DS: [data_point_index, label, [feature]]
# SUM_DS_dict: {label: [sum_feature_i]}
SUM_DS_dict = DS_RDD \
    .map(lambda line: (line[1], line[2])) \
    .reduceByKey(lambda accu, curr: [accu[i] + curr[i] for i in range(len(accu))]) \
    .collectAsMap()
# SUM_DS = sorted(SUM_DS, key = lambda item: item[0])

# DS: [data_point_index, label, [feature]]
# SUMSQ_DS_dict: {label: [sum_feature_i_square]}
SUMSQ_DS_dict = DS_RDD \
    .map(lambda line: (line[1], [feature ** 2 for feature in line[2]])) \
    .reduceByKey(lambda accu, curr: [accu[i] + curr[i] for i in range(len(accu))]) \
    .collectAsMap()
# SUMSQ_DS = sorted(SUMSQ_DS, key = lambda item: item[0])

'''
    Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).
'''
round_0_data_RS_results = KMeans(n_clusters = min(n_cluster * 2, len(pre_RS)), random_state = 1) \
    .fit(get_features_from_line(pre_RS))

CS_labels = sc.parallelize(round_0_data_RS_results.labels_) \
    .map(lambda label: (label, 1)) \
    .groupByKey() \
    .filter(lambda label: len(label[1]) > 1) \
    .map(lambda label: label[0]) \
    .collect()

num_CS_points = len(CS_labels)

RS_labels = sc.parallelize(round_0_data_RS_results.labels_) \
    .map(lambda label: (label, 1)) \
    .groupByKey() \
    .filter(lambda label: len(label[1]) == 1) \
    .map(lambda label: label[0]) \
    .collect()

num_RS_points = len(RS_labels)

# RS_of_index: [data_point_index]
RS_of_index = list()
# RS: [data_point_index, [feature]]
RS = list()
# CS_of_index: [data_point_index]
CS_of_index = list()
# CS: [data_point_index, label, [feature]] -> use `c${label}` for CS labels
CS = list()
# pre_RS: # (0:data_point_index, 1:[feature])
for i in range(len(pre_RS)):
    line = pre_RS[i]
    label = round_0_data_RS_results.labels_[i]
    if label in RS_labels:
        RS_of_index.append(line[0])
        RS.append([line[0], line[1]])
    else:
        CS_of_index.append(line[0])
        CS.append([line[0], "c" + str(label), line[1]])

CS_RDD = sc.parallelize(CS).cache()

# CS_RDD: [0:data_point_index, 1: label, 2:[feature]]
# CS_dict: {label: [(data_point_index, label, [feature])]}
CS_dict = CS_RDD \
    .map(lambda line: (line[1], (line[0], line[2]))) \
    .groupByKey() \
    .map(lambda line: (line[0], list(line[1]))) \
    .collectAsMap()

num_CS_clusters = len(CS_dict.items())

# CS: [data_point_index, cs_label, [feature]]
# N_CS: {cs_label: num of points}
N_CS_dict = CS_RDD \
    .map(lambda line: (line[1], 1)) \
    .reduceByKey(lambda accu, curr: accu + curr) \
    .collectAsMap()

# CS: [data_point_index, cs_label, [feature]]
# SUM_CS: {cs_label, [sum_feature_i]}
SUM_CS_dict = CS_RDD \
    .map(lambda line: (line[1], line[2])) \
    .reduceByKey(lambda accu, curr: [accu[i] + curr[i] for i in range(len(accu))]) \
    .collectAsMap()

# CS: [data_point_index, cs_label, [feature]]
# SUMSQ_CS: {cs_label, [sum_feature_i_square]}
SUMSQ_CS_dict = CS_RDD \
    .map(lambda line: (line[1], [feature ** 2 for feature in line[2]])) \
    .reduceByKey(lambda accu, curr: [accu[i] + curr[i] for i in range(len(accu))]) \
    .collectAsMap()

'''
    the number of the discard points
    the number of the clusters in the compression set
    the number of the compression points
    the number of the points in the retained set
'''
file = open(output_file, 'w')
file.write("The intermediate results:\n")
file.write("Round " + str(round + 1) + ": " + str(num_DS_points) + "," + str(num_CS_clusters) + "," + str(num_CS_points) + "," + str(num_RS_points) + "\n")
round += 1

for i in range(round, NUM_ROUNDS):
    '''
        Step 7. Load another 20% of the data randomly.
    '''
    start_index = i * round_size
    end_index = (i + 1) * round_size
    round_data_RDD = data_RDD \
        .filter(lambda line: start_index <= line[0] and line[0] < end_index)

    '''
        Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign them to the nearest DS clusters if the distance is < 2√d.
    '''
    # round_data: (0:data_point_index, 1:[feature])
    # map: for each points in the load, compute mahalanobis distance to each cluster in DS
    # -> (0:data_point_index, 1:[feature], 2:{label, d})
    # map: for each point's mahalanobis distance, get min mahalanobis distance to a cluster
    # -> (0:data_point_index, 1:[feature], 2:{label, d})
    # map: assign to a cluster in DS or RS
    # -> [0:data_point_index, 1: label, 2:[feature]]
    raw_result_RDD = round_data_RDD \
        .map(lambda line: (line[0], line[1], compute_mahalanobis_distance_between_point_and_cluster(line[1], N_DS_dict, SUM_DS_dict, SUMSQ_DS_dict))) \
        .map(lambda line: (line[0], line[1], get_min_mahalanobis_distance(line[2]))) \
        .map(lambda line: [line[0], assign_DS_label(line[2], DISTANCE_THRESHOLD), line[1]]) \
        .cache()
    
    # Remove unlabeled data points -> label is -1
    DS_RDD = raw_result_RDD \
        .filter(lambda line: line[1] != -1) \
        .cache()

    num_DS_points += DS_RDD.count()

    # Update N, SUM, SUMSQ
    # DS: [data_point_index, label, [feature]]
    # N_DS_dict: {label: num of points}
    new_N_DS_dict = DS_RDD \
        .map(lambda line: (line[1], 1)) \
        .reduceByKey(lambda accu, curr: accu + curr) \
        .collectAsMap()
    
    # DS: [data_point_index, label, [feature]]
    # SUM_DS_dict: {label: [sum_feature_i]}
    new_SUM_DS_dict = DS_RDD \
        .map(lambda line: (line[1], line[2])) \
        .reduceByKey(lambda accu, curr: [accu[i] + curr[i] for i in range(len(accu))]) \
        .collectAsMap()

    # DS: [data_point_index, label, [feature]]
    # SUMSQ_DS_dict: {label: [sum_feature_i_square]}
    new_SUMSQ_DS_dict = DS_RDD \
        .map(lambda line: (line[1], [feature ** 2 for feature in line[2]])) \
        .reduceByKey(lambda accu, curr: [accu[i] + curr[i] for i in range(len(accu))]) \
        .collectAsMap()
    
    # Update N_DS_dict, SUM_DS_dict, SUMSQ_DS_dict
    for label in N_DS_dict.keys():
        N_DS_dict[label] = N_DS_dict[label] + new_N_DS_dict[label]
        SUM_DS_dict[label] = [SUM_DS_dict[label][i] + new_SUM_DS_dict[label][i] for i in range(len(SUM_DS_dict[label]))]
        SUMSQ_DS_dict[label] = [SUMSQ_DS_dict[label][i] + new_SUMSQ_DS_dict[label][i] for i in range(len(SUMSQ_DS_dict[label]))]
    
    DS = DS_RDD.collect()

    for line in DS:
        label = line[1]
        data_point_index = line[0]
        results[data_point_index] = label

    '''
        Step 9. For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and assign the points to the nearest CS clusters if the distance is < 2√d

        for non_DS, split into CS and RS
    '''
    # Remove labeled data points -> label is not -1
    non_DS_RDD = raw_result_RDD \
        .filter(lambda line: line[1] == -1)

    # Remove "-1" label
    # non_DS_RDD: [0:data_point_index, 1: label, 2:[feature]]
    # CS_RDD: [0:data_point_index, 1: label, 2:[feature]]
    non_DS_result_RDD = non_DS_RDD \
        .map(lambda line: (line[0], line[2])) \
        .map(lambda line: (line[0], line[1], compute_mahalanobis_distance_between_point_and_cluster(line[1], N_CS_dict, SUM_CS_dict, SUMSQ_CS_dict))) \
        .map(lambda line: (line[0], line[1], get_min_mahalanobis_distance(line[2]))) \
        .map(lambda line: (line[0], assign_CS_label(line[2], DISTANCE_THRESHOLD), line[1])) \
        .cache()
    
    # CS_RDD: [0:data_point_index, 1: label, 2:[feature]]
    new_CS_RDD = non_DS_result_RDD \
        .filter(lambda line: line[1] != 'r') \
        .cache()
        
    # CS_RDD: [0:data_point_index, 1: label, 2:[feature]]
    # CS_dict: {label: [(data_point_index, label, [feature])]}
    new_CS_dict = new_CS_RDD \
        .map(lambda line: (line[1], [line[0], line[2]])) \
        .groupByKey() \
        .map(lambda line: (line[0], list(line[1]))) \
        .collectAsMap()

    # CS: [data_point_index, cs_label, [feature]]
    # N_CS: {cs_label: num of points}
    new_N_CS_dict = new_CS_RDD \
        .map(lambda line: (line[1], 1)) \
        .reduceByKey(lambda accu, curr: accu + curr) \
        .collectAsMap()

    # CS: [data_point_index, cs_label, [feature]]
    # SUM_CS: {cs_label, [sum_feature_i]}
    new_SUM_CS_dict = new_CS_RDD \
        .map(lambda line: (line[1], line[2])) \
        .reduceByKey(lambda accu, curr: [accu[i] + curr[i] for i in range(len(accu))]) \
        .collectAsMap()

    # CS: [data_point_index, cs_label, [feature]]
    # SUMSQ_CS: {cs_label, [sum_feature_i_square]}
    new_SUMSQ_CS_dict = new_CS_RDD \
        .map(lambda line: (line[1], [feature ** 2 for feature in line[2]])) \
        .reduceByKey(lambda accu, curr: [accu[i] + curr[i] for i in range(len(accu))]) \
        .collectAsMap()
    
    # Update N_CS_dict, SUM_CS_dict, SUMSQ_CS_dict
    for label in new_CS_dict.keys():
        for point in new_CS_dict[label]:
            CS_dict[label].append(point)
        N_CS_dict[label] = N_CS_dict[label] + new_N_CS_dict[label]
        SUM_CS_dict[label] = [SUM_CS_dict[label][i] + new_SUM_CS_dict[label][i] for i in range(len(SUM_CS_dict[label]))]
        SUMSQ_CS_dict[label] = [SUMSQ_CS_dict[label][i] + new_SUMSQ_CS_dict[label][i] for i in range(len(SUMSQ_CS_dict[label]))]
      
    '''
        Step 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
    '''
    # non_DS_RDD: [0:data_point_index, 1: label, 2:[feature]]
    # RS: [0:data_point_index, 1:[feature]]
    RS_RDD = non_DS_result_RDD \
        .filter(lambda line: line[1] == 'r') \
        .map(lambda line: [line[0], line[2]]) \
        .cache()
    
    new_RS = RS_RDD.collect()
    RS = RS + new_RS

    '''
        Step 11. Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).
    '''
    round_data_RS_results = KMeans(n_clusters = min(n_cluster * 2, len(RS)), random_state = 1) \
        .fit(get_features_from_line(RS))
    
    post_CS_labels = sc.parallelize(round_data_RS_results.labels_) \
        .map(lambda label: (label, 1)) \
        .groupByKey() \
        .filter(lambda label: len(label[1]) > 1) \
        .map(lambda label: label[0]) \
        .collect()

    post_RS_labels = sc.parallelize(round_data_RS_results.labels_) \
        .map(lambda label: (label, 1)) \
        .groupByKey() \
        .filter(lambda label: len(label[1]) == 1) \
        .map(lambda label: label[0]) \
        .collect()
    
    # post_CS_of_index: [data_point_index]
    post_CS_of_index = list()
    # post_CS: [data_point_index, label, [feature]] -> use `cc${label}` for CS labels
    post_CS = list()
    # RS: # (0:data_point_index, 1:[feature])
    for i in range(len(RS)):
        line = RS[i]
        label = round_data_RS_results.labels_[i]
        if label in post_RS_labels:
            RS.append([line[0], line[1]])
        else:
            post_CS_of_index.append(line[0])
            post_CS.append([line[0], "pc" + str(label), line[1]])
    
    # Remove point in post_CS in RS
    # RS: [0:data_point_index, 1:[feature]]
    RS = sc.parallelize(RS) \
        .filter(lambda line: line[0] not in post_CS_of_index) \
        .collect()
    
    num_RS_points = len(RS)
    
    post_CS_RDD = sc.parallelize(post_CS).cache()

    # post_CS_RDD: [0:data_point_index, 1: label, 2:[feature]]
    # CS_dict: {label: [(data_point_index, label, [feature])]}
    post_CS_dict = post_CS_RDD \
        .map(lambda line: (line[1], (line[0], line[2]))) \
        .groupByKey() \
        .map(lambda line: (line[0], list(line[1]))) \
        .collectAsMap()
    
    # CS: [data_point_index, cs_label, [feature]]
    # post_N_CS: {cs_label: num of points}
    post_N_CS_dict = post_CS_RDD \
        .map(lambda line: (line[1], 1)) \
        .reduceByKey(lambda accu, curr: accu + curr) \
        .collectAsMap()

    # CS: [data_point_index, cs_label, [feature]]
    # post_SUM_CS: {cs_label, [sum_feature_i]}
    post_SUM_CS_dict = post_CS_RDD \
        .map(lambda line: (line[1], line[2])) \
        .reduceByKey(lambda accu, curr: [accu[i] + curr[i] for i in range(len(accu))]) \
        .collectAsMap()

    # CS: [data_point_index, cs_label, [feature]]
    # post_SUMSQ_CS: {cs_label, [sum_feature_i_square]}
    post_SUMSQ_CS_dict = post_CS_RDD \
        .map(lambda line: (line[1], [feature ** 2 for feature in line[2]])) \
        .reduceByKey(lambda accu, curr: [accu[i] + curr[i] for i in range(len(accu))]) \
        .collectAsMap()

    '''
        Step 12. Merge CS clusters that have a Mahalanobis Distance < 2√𝑑.
        if non-empty CS:
            for pc in post_CS:
                for point in pc
                    merge to closest CS
        else:
            set CS for the first time
    '''
    if CS_dict.keys():
        for label in post_CS_dict.keys():
            post_centroid = [post_SUM_CS_dict[label][i] / post_N_CS_dict[label] for i in range(len(post_SUM_CS_dict[label]))]
            mahalanobis_distance_dict = compute_mahalanobis_distance_between_point_and_cluster(post_centroid, N_CS_dict, SUM_CS_dict, SUMSQ_CS_dict)
            min_mahalanobis_distance = get_min_mahalanobis_distance(mahalanobis_distance_dict)
            # `c${i_prev}` -> `c${i_curr}`
            target_label = min_mahalanobis_distance[0]
            for point in post_CS_dict[label]:
                CS_dict[target_label].append(point)
            N_CS_dict[target_label] = N_CS_dict[target_label] + post_N_CS_dict[label]
            SUM_CS_dict[target_label] = [SUM_CS_dict[target_label][i] + post_SUM_CS_dict[label][i] for i in range(len(SUM_CS_dict[target_label]))]
            SUMSQ_CS_dict[target_label] = [SUMSQ_CS_dict[target_label][i] + post_SUMSQ_CS_dict[label][i] for i in range(len(SUMSQ_CS_dict[target_label]))]
    else:
        for label in post_CS_dict.keys():
            # set `pc${i}` -> `c${i}`
            new_label = label[1:]
            CS_dict[new_label] = post_CS_dict[label]
            N_CS_dict[new_label] = post_N_CS_dict[label]
            SUM_CS_dict[new_label] = post_SUM_CS_dict[label]
            SUMSQ_CS_dict[new_label] = post_SUMSQ_CS_dict[label]
    
    num_CS_clusters = len(CS_dict.items())
    num_CS_points = 0
    for label, count in N_CS_dict.items():
        num_CS_points += count

    '''
        If this is the last run (after the last chunk of data), merge CS clusters with DS clusters that have a Mahalanobis Distance < 2√d.
    '''
    if round == 4:
        if CS_dict.keys():
            for label in CS_dict.keys():
                centroid = [SUM_CS_dict[label][i] / N_CS_dict[label] for i in range(len(SUM_CS_dict[label]))]
                mahalanobis_distance_dict = compute_mahalanobis_distance_between_point_and_cluster(centroid, N_DS_dict, SUM_DS_dict, SUMSQ_DS_dict)
                min_mahalanobis_distance = get_min_mahalanobis_distance(mahalanobis_distance_dict)
                if min_mahalanobis_distance[1] < DISTANCE_THRESHOLD:
                    for point in CS_dict[label]:
                        results[point[0]] = min_mahalanobis_distance[label]
                else:
                    for point in CS_dict[label]:
                        results[point[0]] = -1

        for i in RS:
            results[i[0]] = -1

    '''
        the number of the discard points
        the number of the clusters in the compression set
        the number of the compression points
        the number of the points in the retained set
    '''
    file.write("Round " + str(round + 1) + ": " + str(num_DS_points) + "," + str(num_CS_clusters) + "," + str(num_CS_points) + "," + str(num_RS_points) + "\n")
    round += 1

file.write("\n")
file.write("The clustering results:\n")
for data_point_index in sorted(results.keys()):
    file.write(str(data_point_index) + "," + str(results[data_point_index]) + "\n")
file.close()
print("Duration:", str(time.time() - start_time))