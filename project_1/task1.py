from pyspark import SparkContext
import sys
import json

# A. The total number of reviews
def get_n_review(dataRDD):
  return dataRDD \
    .count()

# B. The number of reviews in 2018
def get_n_review_2018(dataRDD):
  return dataRDD \
    .filter(lambda item: item['date'].split(' ')[0].split('-')[0] == "2018") \
    .count()

# C. The number of distinct users who wrote reviews
def get_n_user(dataRDD):
  return dataRDD \
    .map(lambda item: (item['user_id'], 1)) \
    .distinct() \
    .count()

# D. The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
# sort reduced count desc and sort user_id asc
def get_top10_user(dataRDD):
  return dataRDD \
    .map(lambda item: (item['user_id'], 1)) \
    .reduceByKey(lambda accu, curr: accu + curr) \
    .sortBy(lambda item: (-item[1], item[0])) \
    .take(10)

# E. The number of distinct businesses that have been reviewed
def get_n_business(dataRDD):
  return dataRDD \
    .map(lambda item: (item['business_id'], 1)) \
    .distinct() \
    .count()

# F. The top 10 businesses that had the largest numbers of reviews and the number of reviews they had
# sort reduced count desc and sort business_id asc
def get_top10_business(dataRDD):
  return dataRDD \
    .map(lambda item: (item['business_id'], 1)) \
    .reduceByKey(lambda accu, curr: accu + curr) \
    .sortBy(lambda item: (-item[1], item[0])) \
    .take(10)

sc = SparkContext('local[*]', 'task1')
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
dataRDD = sc.textFile(input_file_path).map(lambda item: json.loads(item))

result = dict()
result['n_review'] = get_n_review(dataRDD)
result['n_review_2018'] = get_n_review_2018(dataRDD)
result['n_user'] = get_n_user(dataRDD)
result['top10_user'] = get_top10_user(dataRDD)
result['n_business'] = get_n_business(dataRDD)
result['top10_business'] = get_top10_business(dataRDD)

with open(output_file_path, 'w') as output_file:
  output_file.write(json.dumps(result))
output_file.close()