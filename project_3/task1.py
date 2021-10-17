from pyspark import SparkContext
import sys
import math
import time
import csv
from itertools import combinations

PARAM_a = [  271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367
                ,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461
                ,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571]

PARAM_b = [  577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661
                ,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773
                ,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883
                ,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997]

NUM_ROWS = 2
NUM_BANDS = 15
THRESHOLD = 0.5

'''
  Map list of user_id to list of indexed user_id with remove duplicates 
'''
def map_user_id_to_index(user_ids, user_dict):
  user_indexes = set()
  for user_id in user_ids:
    user_indexes.add(user_dict[user_id])
  return list(user_indexes)

def generate_signatures(user_indexes, m):
    signatures = []
    num_signatures = NUM_BANDS * NUM_ROWS
    for signature_index in range(num_signatures):
        signature = []
        a = PARAM_a[signature_index % len(PARAM_a)]
        b = PARAM_b[signature_index % len(PARAM_b)]
        for user_index in user_indexes:
            signature.append((a * user_index + b) % m)
        signatures.append(min(signature))
    return signatures


'''
    hash current band into index = map_band_to_bucket()  => bucket index
    take parition of full signature array
'''
def map_band_to_bucket(signature, band_index):
    a = PARAM_a[band_index % len(PARAM_a)]
    b = PARAM_b[band_index % len(PARAM_b)]
    return (a * sum(signature) + b) % 133333333337


def compute_jaccard_similarity(pair, indexed_business_user_dict):
    user_index_list_1 = indexed_business_user_dict[pair[0]]
    user_index_list_2 = indexed_business_user_dict[pair[1]]
    union = len(set(user_index_list_1).union(set(user_index_list_2)))
    intersection = len(set(user_index_list_1).intersection(set(user_index_list_2)))
    return intersection / union


sc = SparkContext('local[*]', 'task1')

start_time = time.time()

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]

data_RDD = sc.textFile(input_file_name)
header = data_RDD.first()
data_RDD = data_RDD.filter(lambda item: item != header)

'''
    Preprocessing users
'''
user_RDD \
  = data_RDD \
      .map(lambda row: row.split(',')[0]) \
      .distinct() \
      .zipWithIndex()
num_user = user_RDD.count()
# dict: {user_id: index}
user_index_dict = user_RDD.collectAsMap()
# dict: {index: user_id}
index_user_dict = {v: k for k, v in user_index_dict.items()}

'''
    Preprocessing businesses
'''
business_RDD \
  = data_RDD \
      .map(lambda row: row.split(',')[1]) \
      .distinct() \
      .zipWithIndex()
num_business = business_RDD.count()
# dict: {business_id: index}
business_index_dict = business_RDD.collectAsMap()
# dict: {index: business_id}
index_business_dict = {v: k for k, v in business_index_dict.items()}

business_user_RDD \
  = data_RDD \
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

# all possible candidate business pairs
candidate_set = set()

for band_index in range(NUM_BANDS):
    # for bucket generated from each set of band, if each bucket has more than 1 band
    # generate a candidate pair for later jc comapre
    # map: cut intervals x[0] and has the interval array of signature
    band_candidate_pairs = business_signature_RDD \
        .map(lambda business_signature: (map_band_to_bucket(business_signature[1][band_index * NUM_ROWS:(band_index + 1) * NUM_ROWS], band_index), business_signature[0])) \
        .groupByKey() \
        .filter(lambda business_band: len(business_band[1]) > 1) \
        .flatMap(lambda business_band: list(combinations(business_band[1], 2))) \
        .collect()
    
    for pair in band_candidate_pairs:
        candidate_set.add(pair)


results = sc.parallelize(list(candidate_set)) \
    .map(lambda pair: (pair, compute_jaccard_similarity(pair, indexed_business_user_dict))) \
    .filter(lambda result: result[1] >= THRESHOLD) \
    .collect()

results = sorted([(sorted([index_business_dict[result[0][0]], index_business_dict[result[0][1]]]), result[1]) for result in results])

with open(output_file_name, 'w') as output_file:
    output_file.write("business_id_1, business_id_2, similarity\n")
    for result in results:
        output_file.write(result[0][0] + "," + result[0][1] + "," + str(result[1]) + "\n")
output_file.close()

print("Duration:", str(time.time() - start_time))


