from pyspark import SparkContext
import sys
import time
import collections
import math
from itertools import combinations

'''
  Give an itemset, generate subset of size from itemset

  itemset: {...{...item}}
'''
def generate_candidate_itemset(itemset, size):
  new_candidates = set()
  for old_candidate_1 in itemset:
      for old_candidate_2 in itemset:
          # (a, b, c) union (a, c, e) => (a, b, c, e) 
          if old_candidate_1 != old_candidate_2:
            candidate = old_candidate_1 | old_candidate_2
            if len(candidate) == size:
                new_candidates.add(candidate)
  return new_candidates

'''
  Count candidate frequent itemset occurrence in all baskets

  baskets: [...(key, (...item))]
  candidate_itemset: {...{...item}}
'''
def count_candidate_itemset(baskets, candidate_itemset):
  # <candidate:frozenset, frequency:int>
  candidate_counts = collections.defaultdict(int)

  # count candidate in current candidate_itemset
  for basket in baskets:
      for candidate in candidate_itemset:
          if candidate.issubset(basket[1]):
            candidate_counts[candidate] += 1

  return candidate_counts

'''
  Filter candidate itemset greater than local support to generate new candidate itemset

  baskets: [...(key, (...item))]
  candidate_counts: {<{...{...item}}>: <occurrence>} -> <candidate_itemset:frozenset, frequency:int>
'''
def filter_candidate_itemset(baskets, candidate_counts, support):
  next_candidate_itemset = set()
  
  # filter out frequent candidates greater than support
  for candidate, count in candidate_counts.items():
      if count >= support:
          next_candidate_itemset.add(candidate)

  return next_candidate_itemset

'''
  A priori with fraction to generate frequent_itemsets on given baskets

  baskets: [...(key, [...item])]
  support: fraction
'''
def generate_frequent_itemsets(baskets, support):
  # generate frequent itemset of size 1 => L1
  singletons = set()
  for basket in baskets:
    for item in basket[1]:
      singletons.add(frozenset({item}))

  candidate_counts = count_candidate_itemset(baskets, singletons)
  singletons = filter_candidate_itemset(baskets, candidate_counts, support)

  result = set()
  
  frequent_item_size = 2
  # C(k) Candidate itemset of size k
  candidate_itemset = singletons
  # L(k) Frequent itemset of size k
  frequent_itemset = singletons

  for item in singletons:
    result.add(item)

  # while frequent itemset is not empty
  while len(frequent_itemset) > 0:
    # Candidates: C(k + 1) = generate candidates with frequent items from last pass => L(k)
    candidate_itemset = generate_candidate_itemset(frequent_itemset, frequent_item_size)
    # Counting: for each basket in baskets, count each candidate's occurrence
    candidate_counts = count_candidate_itemset(baskets, candidate_itemset)
    # Filter: L(k + 1) = set => { candidate | candidate which greater than support }
    frequent_itemset = filter_candidate_itemset(baskets, candidate_counts, support)
    # Save answer: save result of current frequent_item_size for current baskets
    for item in frequent_itemset:
      result.add(item)
    # Go to next iteration
    frequent_item_size += 1

  return result

'''
  Phase 1: Find local candidate itemsets
'''
def find_local_candidate_itemsets(local_baskets, support, num_baskets):
  return generate_frequent_itemsets(local_baskets, (len(local_baskets) / num_baskets) * support)

'''
  Phase 2: Find true frequent itemsets
'''
def count_candidates(local_baskets, candidates):
  # <candidate_itemset:frozenset, frequency:int>
  candidate_counts = collections.defaultdict(int)

  # count candidate in current candidate_itemset
  for basket in local_baskets:
    for candidate in candidates:
        if candidate.issubset(basket[1]):
          candidate_counts[candidate] += 1

  result = set()

  for key, count in candidate_counts.items():
    result.add((key, count))

  return result

def to_string(list):
  if len(list) == 1:
    return '(\'' + list[0] + '\')'
  else:
    res = '(\''
    res += '\', \''.join(list) 
    return res + '\')'

def write_result(output_file, content):
  # sort content
  content_list = list()
  for item in content:
    item = list(item)
    item.sort()
    content_list.append(item)
  content_list.sort(key = lambda item: (len(item), item))

  content_dict = collections.defaultdict(int)

  for item in content_list:
    value = content_dict.get(len(item), list())
    value.append(item)
    content_dict[len(item)] = value
  
  for k, v in content_dict.items():
    line = ''
    for li in v:
      line += to_string(li) + ','
    output_file.write(line[:-1] + '\n\n')

  return


sc = SparkContext('local[*]', 'task2')

filter_threshold = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]
intermediate_file_path = "./intermediate.csv"

start_time = time.time()

'''
  (1) Data preprocessing

  Read the input file and write intermediate file
'''
raw_data_RDD = sc.textFile(input_file_path)
raw_header = raw_data_RDD.first()

# <TRANSACTION_DT-CUSTOMER_ID, PRODUCT_ID>
intermediate_data_RDD = raw_data_RDD \
  .filter(lambda item: item != raw_header) \
  .map(lambda item: ( \
    str(item.split(',')[0].strip("\"")) + "-" + str(item.split(',')[1].strip("\"").lstrip('0')), \
    str(item.split(',')[5].strip("\"").lstrip('0')) \
    ) \
  )
intermediate_data = intermediate_data_RDD.collect()

with open(intermediate_file_path, 'w') as intermediate_file:
  intermediate_file.write("DATE-CUSTOMER_ID,PRODUCT_ID\n")
  for item in intermediate_data:
    intermediate_file.write(item[0] + "," + item[1] + "\n")
intermediate_file.close()

'''
  (2) Apply SON Algorithm: Case 1
  
  Read the intermediate file and write output file
'''
# data_RDD = sc.textFile(input_file_path)
# header = data_RDD.first()
# data_RDD = data_RDD.filter(lambda item: item != header)
# data_RDD = data_RDD.map(lambda item: (str(item.split(',')[0]), str(item.split(',')[1])))

baskets_RDD = intermediate_data_RDD \
  .groupByKey() \
  .map(lambda basket: (basket[0], set(basket[1]))) \
  .filter(lambda basket: len(basket[1]) > filter_threshold)
num_baskets = baskets_RDD.count()

'''
  Phase 1: Find local candidate itemsets

  candidates = candidate_itemset(partition_1) union candidate_itemset(partition_2) ... union candidate_itemset(partition_n)
'''
candidates_RDD = baskets_RDD \
  .mapPartitions(lambda parition_iterator: find_local_candidate_itemsets(list(parition_iterator), support, num_baskets)) \
  .distinct()
candidates = candidates_RDD.collect()

'''
  Phase 2: Find true frequent itemsets
  Go through all baskets, count for each candidate in candidates

  If a candidate count is greater than support, the candidate is true candidate
'''
frequent_itemsets_RDD = baskets_RDD \
  .mapPartitions(lambda parition_iterator: count_candidates(list(parition_iterator), candidates)) \
  .reduceByKey(lambda accu, curr: accu + curr) \
  .filter(lambda candidate: candidate[1] >= support) \
  .map(lambda candidate: candidate[0])
frequent_itemsets = frequent_itemsets_RDD.collect()

with open(output_file_path, 'w') as output_file:
  output_file.write("Candidates:\n")
  write_result(output_file, candidates)
  output_file.write("Frequent Itemsets:\n")
  write_result(output_file, frequent_itemsets)
output_file.close()

print("Duration: " + str(time.time() - start_time))