from pyspark import SparkContext
import sys
import os
import time
import json
import csv
import numpy as np
import xgboost as xgb

'''
Method Description:
In the competition, I implemented the hybrid recommendation system using approach 1, weighted hybrid. There are two parts in the recommendation system, an item-based recommendation system and a model based recommendation system.

In the item-based recommendation system, I compute the prediction by filling out the user-item matrix to get the first results.

In the model-based recommendation system, I used an improved model from assignment 3 to train a XGBregressor to get the predictions. The model uses the following attributes to predict a user's rate on a business:

user.json: extract user’s average_stars, review_count, num_useful, num_funny, num_cool and num_fans

business.json: extract user’s stars, review_count, {attributes}, latitude, longitude

For the attributes, I choose alcohol, caters, RestaurantsPriceRange2, RestaurantsGoodForGroups, RestaurantsTakeOut, RestaurantsDelivery, RestaurantsReservations, GoodForKids, BusinessAcceptsCreditCards, WiFi, Smoking, NoiseLevel, BusinessParking, GoodForMeal

photo.json: extract business’s count of photos in num_food, num_drink, num_inside, num_outside, num_menu

tip.json: extract the number of tips of a user and number of the tips of a business.

In the final combining step, I use the alpha of 0.05, where the weight of the result from an item-based recommendation system is 0.05, and the result from a model-based recommendation system is 0.95. To get a final result, I combine the weighted results.

Error Distribution:
>=0 and <1: 102079
>=1 and <2: 33007
>=2 and <3: 6162
>=3 and <4: 796
>=4: 0

RMSE:
0.9797895095867142

Execution Time:
418.4355640411377s (local machine)
547.9242615699768s (vocareum)

'''
sc = SparkContext('local[*]', 'competition')
sc.setLogLevel("OFF")

start_time = time.time()

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

train_file_name = "yelp_train.csv"
user_file_name = "user.json"
business_file_name = "business.json"
photo_file_name = "photo.json"
tip_file_name = "tip.json"

validation_file_name = "yelp_val.csv"

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

'''
#####################################
#                                   #
# Item-based recommendation system  #
#                                   #
#####################################
'''
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
item_based_predictions = predict_data_RDD \
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

'''
#####################################
#                                   #
# Model-based recommendation system #
#                                   #
#####################################
'''
def get_business_attributes(attributes):
  business_alcohol = 0
  business_caters = 0
  business_RestaurantsPriceRange2 = 0
  business_RestaurantsGoodForGroups = 0
  business_RestaurantsTakeOut = 0
  business_RestaurantsDelivery = 0
  business_RestaurantsReservations = 0
  business_GoodForKids = 0
  business_BusinessAcceptsCreditCards = 0
  business_WiFi = 0
  business_Smoking = 0
  # 1 average, 2: loud, 0: other
  business_NoiseLevel = 0
  business_BusinessParking = 0
  business_GoodForMeal = 0

  if attributes != None:
    if "Alcohol" in attributes and attributes["Alcohol"] != None:
      business_alcohol = 1
    if "Caters" in attributes and attributes["Caters"] != False:
      business_caters = 1
    if "RestaurantsPriceRange2" in attributes:
      business_RestaurantsPriceRange2 = int(attributes["RestaurantsPriceRange2"])
    if "RestaurantsGoodForGroups" in attributes and attributes["RestaurantsGoodForGroups"] != False:
      business_RestaurantsGoodForGroups = 1
    if "RestaurantsTakeOut" in attributes and attributes["RestaurantsTakeOut"] != False:
      business_RestaurantsTakeOut = 1
    if "RestaurantsDelivery" in attributes and attributes["RestaurantsDelivery"] != False:
      business_RestaurantsDelivery = 1
    if "RestaurantsReservations" in attributes and attributes["RestaurantsReservations"] != False:
      business_RestaurantsReservations = 1
    if "GoodForKids" in attributes and attributes["GoodForKids"] != False:
      business_GoodForKids = 1
    if "BusinessAcceptsCreditCards" in attributes and attributes["BusinessAcceptsCreditCards"] != False:
      business_BusinessAcceptsCreditCards = 1
    if "WiFi" in attributes and attributes["WiFi"] == "Free":
      business_WiFi = 1
    if "Smoking" in attributes and attributes["Smoking"] == "yes":
      business_Smoking = 1
    if "NoiseLevel" in attributes:
      if attributes["NoiseLevel"] == "average":
        business_NoiseLevel = 1
      elif attributes["NoiseLevel"] == "loud":
        business_NoiseLevel = 2
    if "BusinessParking" in attributes:
      business_BusinessParking = attributes["BusinessParking"].count("True")
    if "GoodForMeal" in attributes:
      business_GoodForMeal = attributes["GoodForMeal"].count("True")
   
  return [
    business_alcohol,
    business_caters,
    business_RestaurantsPriceRange2,
    business_RestaurantsGoodForGroups,
    business_RestaurantsTakeOut,
    business_RestaurantsDelivery,
    business_RestaurantsReservations,
    business_GoodForKids,
    business_BusinessAcceptsCreditCards,
    business_WiFi,
    business_Smoking,
    business_NoiseLevel,
    business_BusinessParking,
    business_GoodForMeal
  ]

def generate_CF_attribute_vector(
      pair, 
      user_dict, 
      business_dict, 
      user_index_dict, 
      business_index_dict, 
      business_photo_count_dict,
      user_tip_dict,
      business_tip_dict
    ):
  user_id = pair[0]
  business_id = pair[1]

  user_rate_average = 0.0
  business_rate_average = 0.0
  user_review_count = 0
  business_review_count = 0
  user_useful = 0
  user_funny = 0
  user_cool = 0
  user_fans = 0
  business_photos = [0, 0, 0, 0, 0]
  user_tips = 0
  business_tips = 0
  business_latitude = -1
  business_longitude = -1
  
  if user_id in user_dict.keys():
    user_rate_average = user_dict[user_id][0]
    user_review_count = user_dict[user_id][1]
    user_useful = user_dict[user_id][2]
    user_funny = user_dict[user_id][3]
    user_cool = user_dict[user_id][4]
    user_fans = user_dict[user_id][5]

  if business_id in business_dict.keys():
    business_rate_average = business_dict[business_id][0]
    business_review_count = business_dict[business_id][1]
    business_attributes = get_business_attributes(business_dict[business_id][2])
    business_latitude = business_dict[business_id][3]
    business_longitude = business_dict[business_id][4]
  
  if business_id in business_photo_count_dict.keys():   
    business_photos = business_photo_count_dict[business_id]
  
  if user_id in user_tip_dict.keys():
    user_tips = user_tip_dict[user_id]
  
  if business_id in business_tip_dict.keys():
    business_tips = business_tip_dict[business_id]
  
  return [
      user_index_dict[user_id], 
      user_rate_average, 
      business_index_dict[business_id],
      business_rate_average, 
      user_review_count, 
      business_review_count,
      user_useful,
      user_funny,
      user_cool,
      user_fans,
      business_attributes[0],
      business_attributes[1],
      business_attributes[2],
      business_attributes[3],
      business_attributes[4],
      business_attributes[5],
      business_attributes[6],
      business_attributes[7],
      business_attributes[8],
      business_attributes[9],
      business_attributes[10],
      business_photos[0],
      business_photos[1],
      business_photos[2],
      business_photos[3],
      business_photos[4],
      user_tips,
      business_tips,
      business_latitude,
      business_longitude,
    ]

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
index_user_dict = {v: k for k, v in user_index_dict.items()}

# {business_id: index}
business_index_dict = train_business_RDD.union(predict_business_RDD) \
  .distinct() \
  .zipWithIndex() \
  .collectAsMap()
# {index: business_id}
index_business_dict = {v: k for k, v in business_index_dict.items()}

'''
  Pre-precessing data for classifier
  user.json
  -> user_dict: {user_id: [average_stars, review_count, useful, funny, cool, fans]}

  business.json
  -> business_dict:  {business_id: [stars, review_count, {attributes}, latitude, longitude]}

  photo.json
  -> business_photo_label_dict: {business_id: [num_food, num_drink, num_inside, num_outside, num_menu]}

  tip.json
  -> user_tip_dict: {user_id: tip_count}
  -> business_tip_dict: {business_id: tip_count}
'''
user_RDD = sc.textFile(os.path.join(folder_path, user_file_name)).map(lambda line: json.loads(line))
user_dict = user_RDD \
  .map(lambda user: (
    user["user_id"], 
    [
      user["average_stars"], 
      user["review_count"], 
      user["useful"],
      user["funny"],
      user["cool"],
      user["fans"],
    ])) \
  .collectAsMap()

business_RDD = sc.textFile(os.path.join(folder_path, business_file_name)).map(lambda line: json.loads(line))
business_dict = business_RDD \
  .map(lambda business: (
    business["business_id"], 
    [
      business["stars"], 
      business["review_count"],
      business["attributes"],
      business["latitude"],
      business["longitude"]
    ])) \
  .collectAsMap()

photo_data_RDD = sc.textFile(os.path.join(folder_path, photo_file_name)).map(lambda line: json.loads(line))
business_photo_label_dict = photo_data_RDD \
  .map(lambda item: (item["business_id"], item["label"])) \
  .groupByKey() \
  .collectAsMap()
business_photo_count_dict = dict()
for k, v in business_photo_label_dict.items():
  num_food = 0
  num_drink = 0
  num_inside = 0
  num_outside = 0
  num_menu = 0
  for label in v:
    if label == 'food':
      num_food += 1
    elif label == 'drink':
      num_drink += 1
    elif label == 'inside':
      num_inside += 1
    elif label == 'outside':
      num_outside += 1
    elif label == 'menu':
      num_menu += 1
  business_photo_count_dict[k] = [num_food, num_drink, num_inside, num_outside, num_menu]

tip_data_RDD = sc.textFile(os.path.join(folder_path, tip_file_name)).map(lambda line: json.loads(line))
user_tip_dict = tip_data_RDD \
  .map(lambda item: (item["user_id"], 1)) \
  .groupByKey() \
  .map(lambda item: (item[0], len(item[1]))) \
  .collectAsMap()
business_tip_dict = tip_data_RDD \
  .map(lambda item: (item["business_id"], 1)) \
  .groupByKey() \
  .map(lambda item: (item[0], len(item[1]))) \
  .collectAsMap()

'''
  Transform train_data_RDD, predict_data_RDD, user_dict, business_dict into train_attributes_array
  train_attributes_array:    [user_id, user_rate_average, business_id, business_rate_average, ...other_features]
  predict_attributes_array:    [user_id, user_rate_average, business_id, business_rate_average, ...other_features]

  Transform train_data_RDD.csv into train_labels_array
  train_labels_array:        [rate]
'''
# 0: user_id, 1: business_id, 2: rate
# [user_id, user_rate_average, business_id, business_rate_average, ...other_features]
train_attributes = train_data_RDD \
  .map(lambda item: generate_CF_attribute_vector(
    pair = item,
    user_dict = user_dict,
    business_dict = business_dict,
    user_index_dict = user_index_dict,
    business_index_dict = business_index_dict,
    business_photo_count_dict = business_photo_count_dict,
    user_tip_dict = user_tip_dict,
    business_tip_dict = business_tip_dict
  )) \
  .collect()

# 0: user_id, 1: business_id, 2: rate
# rate
train_labels = train_data_RDD \
  .map(lambda item: item[2]) \
  .collect()

# 0: user_id, 1: business_id, 2: rate
# [user_id, user_rate_average, business_id, business_rate_average, ...other_features]
predict_attributes = predict_data_RDD \
  .map(lambda item: generate_CF_attribute_vector(
    pair = item,
    user_dict = user_dict,
    business_dict = business_dict,
    user_index_dict = user_index_dict,
    business_index_dict = business_index_dict,
    business_photo_count_dict = business_photo_count_dict,
    user_tip_dict = user_tip_dict,
    business_tip_dict = business_tip_dict
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
      ...other_features -user.json, business.json, photo.json, tip.json
    ) -> rate

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
# [user_avg_rate, user_review_count, ...other_features] -> rate
xgb_regressor.fit(train_attributes_array, train_labels_array)
predict_labels_array = xgb_regressor.predict(predict_attributes_array)

'''
#####################################
#                                   #
# Hybrid recommendation system      #
#                                   #
#####################################
'''
alpha = 0.05

with open(output_file_name, 'w') as csv_file:
  csv_writer = csv.writer(csv_file, delimiter=',')
  csv_writer.writerow(["user_id", "business_id", "prediction"])
  for i in range(len(predict_labels_array)):
      csv_writer.writerow([
        index_user_dict[predict_attributes[i][0]], 
        index_business_dict[predict_attributes[i][2]], 
        alpha * float(item_based_predictions[i][2]) + 
          (1 - alpha) * float(predict_labels_array[i])
      ])
csv_file.close()

duration = str(time.time() - start_time)

'''
  Validation result
'''  
with open(os.path.join(folder_path, validation_file_name)) as validation_file:
    ground_truth = validation_file.readlines()[1:]

error_distribution_dict = {
  ">=0 and <1": 0, 
  ">=1 and <2": 0, 
  ">=2 and <3": 0, 
  ">=3 and <4": 0, 
  ">=4": 0
}

RMSE = 0
for i in range(len(predict_labels_array)):
  prediction = alpha * float(item_based_predictions[i][2]) + (1 - alpha) * float(predict_labels_array[i])
  diff = abs(prediction - float(ground_truth[i].split(",")[2]))
  RMSE += diff ** 2
  if diff < 1:
      error_distribution_dict[">=0 and <1"] = error_distribution_dict[">=0 and <1"] + 1
  elif 2 > diff >= 1:
      error_distribution_dict[">=1 and <2"] = error_distribution_dict[">=1 and <2"] + 1
  elif 3 > diff >= 2:
      error_distribution_dict[">=2 and <3"] = error_distribution_dict[">=2 and <3"] + 1
  elif 4 > diff >= 3:
      error_distribution_dict[">=3 and <4"] = error_distribution_dict[">=3 and <4"] + 1
  else:
      error_distribution_dict[">=4"] = error_distribution_dict[">=4"] + 1

RMSE = (RMSE/len(predict_labels_array)) ** (1/2)

print("Error Distribution:")
for k, v in error_distribution_dict.items():
  print(k + ":", v)
print()
print("RMSE:")
print(str(RMSE))
print()
print("Duration:")
print(str(duration))

