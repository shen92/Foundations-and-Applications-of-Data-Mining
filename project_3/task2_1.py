from pyspark import SparkContext
import sys
import time
import csv

'''
  Merge two dicts
'''
def add_to_dict(curr_dict, new_dict):
    curr_dict.update(new_dict)
    return curr_dict

'''
  Compute Pearson Correlation between two businesses
'''
def compute_pearson_correlation_between_business(
      business_id_1, 
      business_id_2,
      business_user_rate_dict
    ):
  if business_id_1 == business_id_2:
    return (business_id_2, 0)
  
  if business_id_1 not in business_rate_average_dict.keys() or business_id_2 not in business_rate_average_dict.keys() :
    return (business_id_2, 0)
  
  user_rate_business_1 = business_user_rate_dict[business_id_1]
  user_rate_business_2 = business_user_rate_dict[business_id_2]
  common_user_indexes = set(user_rate_business_1.keys()).intersection(set(user_rate_business_2.keys()))

  if len(common_user_indexes) == 0:
    return (business_id_2, 0)
  
  sum_business_1 = 0
  sum_business_2 = 0

  # Compute average for co-rated items
  for user_index in common_user_indexes:
    sum_business_1 += user_rate_business_1[user_index]
    sum_business_1 += user_rate_business_2[user_index]

  avg_business_1 = sum_business_1 / len(common_user_indexes)
  avg_business_2 = sum_business_2 / len(common_user_indexes)
  
  numerator = 0
  variance_business_1 = 0
  variance_business_2 = 0

  # Compute Pearson Correlation
  for user_index in common_user_indexes:
    numerator += (user_rate_business_1[user_index] - avg_business_1) * (user_rate_business_2[user_index] - avg_business_2)
    variance_business_1 += (user_rate_business_1[user_index] - avg_business_1) ** 2
    variance_business_2 += (user_rate_business_2[user_index] - avg_business_2) ** 2
  
  pearson_correlation = 0

  if variance_business_1 != 0 and variance_business_2 != 0:
    pearson_correlation = numerator / (variance_business_1 * variance_business_2) ** 0.5
  
  return (business_id_2, pearson_correlation)

'''
  Compute prediction rate for (user_id, business_id)
'''
def compute_prediction_rate(
      user_id, business_id, 
      pearson_correlations, 
      business_user_rate_dict, 
      user_business_rate_dict, 
      business_rate_average_dict,
      user_rate_average_dict
  ):
  pearson_correlations = list(filter(lambda item: item[1] > 0, pearson_correlations))

  weighted_rate_sum = 0
  weight_sum = 0

  for pearson_correlation in pearson_correlations:
    if pearson_correlation[0] in user_business_rate_dict.keys():
      neighbor_item_rate = user_business_rate_dict[pearson_correlation[0]]
      weighted_rate_sum += pearson_correlation[1] * neighbor_item_rate
      weight_sum += pearson_correlation[1]
  
  rate = 0

  if weight_sum > 0:
    rate = weighted_rate_sum / weight_sum
  else:
    if business_id in business_rate_average_dict.keys():
      rate = business_rate_average_dict[business_id]
    elif user_id in user_rate_average_dict.keys():
      rate = user_rate_average_dict[user_id]
    else:
      rate = 3.0
  
  return rate

'''
  Give a key of user_id, retrive its value of {bussiness_id:rate}
'''
def get_user_business_ids(user_id, user_business_rate_dict):
  if user_id in user_business_rate_dict.keys():
    return user_business_rate_dict[user_id]
  return {}

sc = SparkContext('local[*]', 'task2_1')

start_time = time.time()

# user-item matrix
train_file_name = sys.argv[1]
# user-item pairs to predict
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

# user_id, business_id, stars
train_data_RDD = sc.textFile(train_file_name)
header = train_data_RDD.first()
train_data_RDD = train_data_RDD.filter(lambda item: item != header and len(item) > 0)
# 0: user_id, 1: business_id, 2: rate
train_data_RDD = train_data_RDD.map(lambda item: (item.split(',')[0], item.split(',')[1], float(item.split(',')[2])))

# user_id, business_id, stars
predict_data_RDD = sc.textFile(test_file_name)
header = predict_data_RDD.first()
predict_data_RDD = predict_data_RDD.filter(lambda item: item != header and len(item) > 0)
# 0: user_id, 1: business_id
predict_data_RDD = predict_data_RDD.map(lambda item: (item.split(',')[0], item.split(',')[1]))

'''
  Create business set
'''
businesses = train_data_RDD \
  .map(lambda item: item[1]) \
  .distinct() \
  .collect()

'''
  Create {user_id:{business_id:rate}} dict
  user_business_rate_dict: {user_id: [{business_id: rate}]}
  1)  map: (0:user_id, (1:business_id, 2:rate))
  2)  reduceByKey: (user_id, {business_id: rate})
'''
user_business_rate_dict = train_data_RDD \
  .map(lambda item: (item[0], {item[1]: item[2]})) \
  .reduceByKey(lambda accu, curr: add_to_dict(accu, curr)) \
  .collectAsMap()

'''
  Create {business_id:{user_id:rate}} dict
  business_user_rate_dict: {business_id: [{user_id: rate}]}
  1)  map: (1:business_id, (0:user_id, 2:rate))
  2)  reduceByKey: (business_id, {user_id, rate})
'''
business_user_rate_dict = train_data_RDD \
  .map(lambda item: (item[1], {item[0]: item[2]})) \
  .reduceByKey(lambda accu, curr: add_to_dict(accu, curr)) \
  .collectAsMap()

'''
  Create {business_id:average_rate} dict for fast access
  1)  map: (1: business_id, 2: rate)
  2)  groupByKey: (business_id, [rate])
  3)  map: (business_id, avg([rate]))
'''
business_rate_average_dict = train_data_RDD \
  .map(lambda item: (item[1], item[2])) \
  .groupByKey() \
  .map(lambda item: (item[0], sum(item[1]) / len(item[1]))) \
  .collectAsMap()

'''
  Create {user:average_rate} dict for fast access
  1)  map: (0: user_id, 2: rate)
  2)  groupByKey: (user_id, [rate])
  3)  map: (user_id, avg([rate]))
'''
user_rate_average_dict = train_data_RDD \
  .map(lambda item: (item[0], item[2])) \
  .groupByKey() \
  .map(lambda item: (item[0], sum(item[1]) / len(item[1]))) \
  .collectAsMap()

'''
  For each (0: user_id, 1: business_id) pair,
  1)  compute all similarities from item_1 to item_n, except itself
      map: (0: user_id, 1: business_id, [adjacent_pearson_correlation])
  2)  compute prediction rate
      map: (0: user_id, 1: business_id, prediction_rate)
'''
results = predict_data_RDD \
  .map(lambda item: (
      item[0], 
      item[1],
      [compute_pearson_correlation_between_business(
        business_id_1 = item[1], 
        business_id_2 = business_id,
        business_user_rate_dict = business_user_rate_dict
      ) for business_id in get_user_business_ids(item[0], user_business_rate_dict).keys()]
  )) \
  .map(lambda item: (
    item[0], 
    item[1],
    compute_prediction_rate(
      user_id = item[0],
      business_id = item[1],
      pearson_correlations = item[2],
      business_user_rate_dict = business_user_rate_dict,
      user_business_rate_dict = get_user_business_ids(item[0], user_business_rate_dict),
      business_rate_average_dict = business_rate_average_dict,
      user_rate_average_dict = user_rate_average_dict
    ) 
  )) \
  .collect()

with open(output_file_name, 'w') as csv_file:
  csv_writer = csv.writer(csv_file, delimiter=',')
  csv_writer.writerow(["user_id", "business_id", "prediction"])
  for result in results:
      csv_writer.writerow([result[0], result[1], str(float(result[2]))])
csv_file.close()

print("Duration:", str(time.time() - start_time))