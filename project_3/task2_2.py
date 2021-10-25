from pyspark import SparkContext
import sys
import os
import time
import json
import csv
import numpy as np
import xgboost as xgb

sc = SparkContext('local[*]', 'task2_2')
sc.setLogLevel("OFF")

start_time = time.time()

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

train_file_name = "yelp_train.csv"
user_file_name = "user.json"
business_file_name = "business.json"
# review_file_name = "review_train.json"

'''
  Pre-processing input and output data

  train_data_RDD:       (0: user_id, 1: business_id, 2: rate)
  -> train_user_RDD: (0: user_id)
  -> train_business_RDD: (1: business_id)

  predict_data_RDD:  (0: user_id, 1: business_id)
  -> predict_user_RDD: (0: user_id)
  -> predict_business_RDD: (1: business_id)
'''
# user_id, business_id, stars
train_data_RDD = sc.textFile(os.path.join(folder_path, train_file_name))
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

train_user_RDD = train_data_RDD \
  .map(lambda item: item[0])
train_business_RDD = train_data_RDD \
  .map(lambda item: item[1])

predict_user_RDD = predict_data_RDD \
  .map(lambda item: item[0])
predict_business_RDD = predict_data_RDD \
  .map(lambda item: item[1])

# {user_id: index}
user_index_dict = train_user_RDD.union(predict_user_RDD) \
  .distinct() \
  .zipWithIndex() \
  .collectAsMap()
# {index: user_id}
index_user_dict = { index: user_id for user_id, index in user_index_dict.items() }

# {business_id: index}
business_index_dict = train_business_RDD.union(predict_business_RDD) \
  .distinct() \
  .zipWithIndex() \
  .collectAsMap()
# {index: business_id}
index_business_dict = { index: business_id for business_id, index in business_index_dict.items() }

'''
  Pre-precessing user.json and business.json
  user.json   [user_id, average_stars, review_count] 
  -> user_dict: {user_id: [average_stars, review_count]}

  business.json
  [business_id, stars, review_count] 
  -> business_dict:  {business_id: [stars, review_count]}
'''
user_RDD = sc.textFile(os.path.join(folder_path, user_file_name)).map(lambda line: json.loads(line))
business_RDD = sc.textFile(os.path.join(folder_path, business_file_name)).map(lambda line: json.loads(line))

user_dict = user_RDD \
  .map(lambda user: (user["user_id"], [user["average_stars"], user["review_count"]])) \
  .collectAsMap()

business_dict = business_RDD \
  .map(lambda business: (business["business_id"], [business["stars"], business["review_count"]])) \
  .collectAsMap()

'''
  Transform train_data_RDD, predict_data_RDD, user_dict, business_dict into train_attributes_array
  train_attributes_array:    [user_id, user_rate_average, business_id, business_rate_average, *user_review_count, *business_review_count]
  predict_attributes_array:    [user_id, user_rate_average, business_id, business_rate_average, *user_review_count, *business_review_count]

  Transform train_data_RDD.csv into train_labels_array
  train_labels_array:        [rate]
'''

def generate_CF_attribute_vector(pair, user_dict, business_dict, user_index_dict, business_index_dict):
  user_id = pair[0]
  business_id = pair[1]

  user_rate_average = 0.0
  business_rate_average = 0.0
  user_review_count = 0
  business_review_count = 0

  if user_id in user_dict.keys():
    user_rate_average = user_dict[user_id][0]
    user_review_count = user_dict[user_id][1]

  if business_id in business_dict.keys():
    business_rate_average = business_dict[business_id][0]
    business_review_count = business_dict[business_id][1]
   
  return [user_index_dict[user_id], user_rate_average, business_index_dict[business_id], business_rate_average, user_review_count, business_review_count]

# 0: user_id, 1: business_id, 2: rate
# [user_id, user_rate_average, business_id, business_rate_average, *user_review_count, *business_review_count]
train_attributes = train_data_RDD \
  .map(lambda item: generate_CF_attribute_vector(
    pair = item,
    user_dict = user_dict,
    business_dict = business_dict,
    user_index_dict = user_index_dict,
    business_index_dict = business_index_dict
  )) \
  .collect()

# 0: user_id, 1: business_id, 2: rate
# rate
train_labels = train_data_RDD \
  .map(lambda item: item[2]) \
  .collect()

# 0: user_id, 1: business_id, 2: rate
# [user_id, user_rate_average, business_id, business_rate_average, *user_review_count, *business_review_count]
predict_attributes = predict_data_RDD \
  .map(lambda item: generate_CF_attribute_vector(
    pair = item,
    user_dict = user_dict,
    business_dict = business_dict,
    user_index_dict = user_index_dict,
    business_index_dict = business_index_dict
  )) \
  .collect()

train_attributes_array = np.array(train_attributes)
train_labels_array = np.array(train_labels)
predict_attributes_array = np.array(predict_attributes)

'''
  Classifier Model for prediction: 
    f(user_id,        - user.json / yelp_train.csv
      average_stars,  - user.json
      business_id,    - business.json / yelp_train.csv
      stars,          - business.json
      *review_count,  - user.json
      *review_count   - business.json
    ) -> rate
  * for optional

  Train:    train_attributes_array -> train_labels_array
  Predict:  predict_attributes_array -> predict_labels_array
'''
xgb_regressor = xgb.XGBRegressor(
  verbosity = 0,
  objective ='reg:linear',
  learning_rate = 0.1,
  max_depth = 7,
  n_estimators = 90,
  subsample = 0.5,
  random_state = 1
)

# [user_avg_rate, user_review_count, business_average_star, business_count] -> rate
xgb_regressor.fit(train_attributes_array, train_labels_array)

predict_labels_array = xgb_regressor.predict(predict_attributes_array)

with open(output_file_name, 'w') as csv_file:
  csv_writer = csv.writer(csv_file, delimiter=',')
  csv_writer.writerow(["user_id", "business_id", "prediction"])
  for i in range(len(predict_labels_array)):
      csv_writer.writerow([index_user_dict[predict_attributes[i][0]], index_business_dict[predict_attributes[i][2]], predict_labels_array[i]])
csv_file.close()

print("Duration:", str(time.time() - start_time))