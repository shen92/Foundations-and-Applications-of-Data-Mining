from pyspark import SparkContext
import sys
import json
import time

# city_average_stars 
#   => reduceByKey(k: city,  v: (star, count)) 
#   => (k: city, v: (sum(star), sum(count)))
#   => (k: city, v: (sum(star) / sum(count)))
def get_city_average_stars(reviewRDD, businessRDD):
  return reviewRDD.join(businessRDD) \
    .map(lambda item: (item[1][1], (item[1][0], 1))) \
    .reduceByKey(lambda accu, curr: (accu[0] + curr[0], accu[1] + curr[1])) \
    .map(lambda item: (item[0], (item[1][0] / item[1][1]))) \
    
def get_top10_with_spark(reviewRDD, businessRDD):
  start_time = time.time()

  top10_city_average_stars = get_city_average_stars(reviewRDD, businessRDD) \
    .sortBy(lambda item: (-item[1], item[0])) \
    .take(10)
  
  return time.time() - start_time

def get_top10_with_python(reviewRDD, businessRDD):
  start_time = time.time()

  city_average_stars = get_city_average_stars(reviewRDD, businessRDD).collect()
  top10_city_average_stars = sorted(city_average_stars, key=lambda item: (-item[1], item[0]))[0:10]

  return time.time() - start_time

sc = SparkContext('local[*]', 'task3')
review_filepath = sys.argv[1]
business_filepath = sys.argv[2]
output_filepath_question_a = sys.argv[3]
output_filepath_question_b = sys.argv[4]

reviewRDD = sc.textFile(review_filepath) \
  .map(lambda item: json.loads(item)) \
  .map(lambda review: (review['business_id'], review['stars']))

businessRDD = sc.textFile(business_filepath) \
  .map(lambda item: json.loads(item)) \
  .map(lambda business: (business['business_id'], business['city']))

# Task 4.a
result_a = get_city_average_stars(reviewRDD, businessRDD) \
            .sortBy(lambda item: (-item[1], item[0])) \
            .collect()

# Task 4.b
result_b = dict()
# Piazza: You need to rerun the remaining part except for the init process (argument parsing, data reading, etc.)
# Method1: Collect all the data, sort in python, and then print the first 10 cities
result_b['m1'] = get_top10_with_python(reviewRDD, businessRDD)
# Method2: Sort in Spark, take the first 10 cities, and then print these 10 cities
result_b['m2'] = get_top10_with_spark(reviewRDD, businessRDD)
result_b['reason'] = "With spark, we sort the top 10 cities with highest stars in parallel. Without spark, we load all data in to memory and process all. With spark, the map, shuffle and reduce operation of sort take time while python almost doesn't. The performance is better with python with relative small inputs, and with scale of input grows, the performace with spark is better (may not fit in memeory for python if dataset is too large)."

with open(output_filepath_question_a, 'w') as output_file_question_a:
  output_file_question_a.write("city,stars\n")
  for row in result_a:
    output_file_question_a.write(row[0] + "," + str(row[1]) + "\n")
output_file_question_a.close()

with open(output_filepath_question_b, 'w') as output_file_question_b:
  output_file_question_b.write(json.dumps(result_b, indent=2))
output_file_question_b.close()