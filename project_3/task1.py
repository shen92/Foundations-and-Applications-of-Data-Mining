from pyspark import SparkContext
import sys
import time
from itertools import combinations
import csv

PARAM_a = [ 30011, 30013, 30029, 30047, 30059, 30071, 30089, 30091, 30097, 30103,
            30109, 30113, 30119, 30133, 30137, 30139, 30161, 30169, 30181, 30187,
            30197, 30203, 30211, 30223, 30241, 30253, 30259, 30269, 30271, 30293,
            30307, 30313, 30319, 30323, 30341, 30347, 30367, 30389, 30391, 30403,
            30427, 30431, 30449, 30467, 30469, 30491, 30493, 30497, 30509, 30517,
            30529, 30539, 30553, 30557, 30559, 30577, 30593, 30631, 30637, 30643,
            30649, 30661, 30671, 30677, 30689, 30697, 30703, 30707, 30713, 30727,
            30757, 30763, 30773, 30781, 30803, 30809, 30817, 30829, 30839, 30841,
            30851, 30853, 30859, 30869, 30871, 30881, 30893, 30911, 30931, 30937,
            30941, 30949, 30971, 30977, 30983, 31013, 31019, 31033, 31039, 31051,
            31063, 31069, 31079, 31081, 31091, 31121, 31123, 31139, 31147, 31151,
            31153, 31159, 31177, 31181, 31183, 31189, 31193, 31219, 31223, 31231,
            31237, 31247, 31249, 31253, 31259, 31267, 31271, 31277, 31307, 31319 ]

PARAM_b = [ 71263, 71287, 71293, 71317, 71327, 71329, 71333, 71339, 71341, 71347,
            71353, 71359, 71363, 71387, 71389, 71399, 71411, 71413, 71419, 71429,
            71437, 71443, 71453, 71471, 71473, 71479, 71483, 71503, 71527, 71537,
            71549, 71551, 71563, 71569, 71593, 71597, 71633, 71647, 71663, 71671,
            71693, 71699, 71707, 71711, 71713, 71719, 71741, 71761, 71777, 71789,
            71807, 71809, 71821, 71837, 71843, 71849, 71861, 71867, 71879, 71881 ]

NUM_ROWS = 2
NUM_BANDS = 16
THRESHOLD = 0.5
M = 50287
P = 92143

'''
  Map list of user_id to list of indexed user_id with remove duplicates 
'''
def map_user_id_to_index(user_ids, user_dict):
  user_indexes = set()
  for user_id in user_ids:
    user_indexes.add(user_dict[user_id])
  return list(user_indexes)

'''
    Generate signature for indexed business:[user] -> [user]
'''
def generate_signatures(user_indexes, m):
    signatures = []
    num_signatures = NUM_BANDS * NUM_ROWS
    for signature_index in range(num_signatures):
        signature = []
        a = PARAM_a[signature_index % len(PARAM_a)]
        b = PARAM_b[signature_index + 1 % len(PARAM_b)]
        for user_index in user_indexes:
            signature.append((a * user_index + b) % P % m)
        signatures.append(min(signature))
    return signatures

'''
    hash current band into index = hash_band_to_bucket()  => bucket index
    take parition of full signature array
'''
def hash_band_to_bucket(signature, band_index):
    a = PARAM_a[band_index % len(PARAM_a)]
    b = PARAM_b[band_index + 1 % len(PARAM_b)]
    return (b * sum(signature) + a) % P % M

'''
    comapre original business:[user] (user list) jaccard similarity
'''
def compute_jaccard_similarity(pair, indexed_business_user_dict):
    user_index_list_1 = indexed_business_user_dict[pair[0]]
    user_index_list_2 = indexed_business_user_dict[pair[1]]
    return len(set(user_index_list_1).intersection(set(user_index_list_2))) / len(set(user_index_list_1).union(set(user_index_list_2)))


sc = SparkContext('local[*]', 'task1')
sc.setLogLevel("OFF")

start_time = time.time()

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]

data_RDD = sc.textFile(input_file_name)
header = data_RDD.first()
data_RDD = data_RDD.filter(lambda item: item != header)

'''
    Preprocessing users
'''
user_RDD = data_RDD \
    .map(lambda row: row.split(',')[0]) \
    .distinct() \
    .zipWithIndex()
num_user = user_RDD.count()
# dict: {user_id: index}
user_index_dict = user_RDD.collectAsMap()
# dict: {index: user_id}
index_user_dict = { index: user_id for user_id, index in user_index_dict.items() }

'''
    Preprocessing businesses
'''
business_RDD = data_RDD \
    .map(lambda row: row.split(',')[1]) \
    .distinct() \
    .zipWithIndex()
num_business = business_RDD.count()
# dict: {business_id: index}
business_index_dict = business_RDD.collectAsMap()
# dict: {index: business_id}
index_business_dict = { index: business_id for business_id, index in business_index_dict.items() }

'''
    Generate signature matrix
'''
business_user_RDD = data_RDD \
    .map(lambda row: (row.split(',')[1], row.split(',')[0])) \
    .groupByKey() \
    .map(lambda item: (item[0], list(item[1])))
# list: (business_id: [user_id])
business_user = business_user_RDD.collect()
business_user_dict = business_user_RDD.collectAsMap()
indexed_business_user_RDD \
    = business_user_RDD.map(lambda business_user: (business_index_dict[business_user[0]], map_user_id_to_index(business_user[1], user_index_dict)))
indexed_business_user_dict = indexed_business_user_RDD.collectAsMap()

business_signature_RDD \
    = indexed_business_user_RDD.map(lambda business_user: (business_user[0], generate_signatures(business_user[1], num_user)))

'''
    Hash each band into buckets
'''
# all possible candidate business pairs
candidate_set = set()

for band_index in range(NUM_BANDS):
    # for bucket generated from each set of band, if each bucket has more than 1 band
    # generate a candidate pair for later jc comapre
    # map: cut intervals x[0] and has the interval array of signature
    band_candidate_pairs = business_signature_RDD \
        .map(lambda business_signature: (hash_band_to_bucket(business_signature[1][band_index * NUM_ROWS:(band_index + 1) * NUM_ROWS], band_index), business_signature[0])) \
        .groupByKey() \
        .filter(lambda business_band: len(business_band[1]) > 1) \
        .flatMap(lambda business_band: list(combinations(business_band[1], 2))) \
        .collect()
    
    for pair in band_candidate_pairs:
        candidate_set.add(pair)

'''
    Verify candidate pairs
'''
results = sc.parallelize(candidate_set) \
    .map(lambda pair: (pair, compute_jaccard_similarity(pair, indexed_business_user_dict))) \
    .filter(lambda result: result[1] >= THRESHOLD) \
    .collect()

results = [(sorted([index_business_dict[result[0][0]], index_business_dict[result[0][1]]]), result[1]) for result in results]
results = sorted(results)

with open(output_file_name, 'w') as csv_file:
  csv_writer = csv.writer(csv_file, delimiter=',')
  csv_writer.writerow(["business_id_1", "business_id_2", "similarity"])
  for result in results:
      csv_writer.writerow([result[0][0], result[0][1], str(result[1])])
csv_file.close()

print("Duration:", str(time.time() - start_time))