from pyspark import SparkContext
import sys
import json
import time

# F. The top 10 businesses that had the largest numbers of reviews and the number of reviews they had
# sort reduced count desc and sort business_id asc
def get_top10_business_analysis(dataRDD):
  res = dict()
  res['n_partition'] = dataRDD.getNumPartitions()
  res['n_items'] = dataRDD.glom().map(lambda partition: len(partition)).collect()
  start_time = time.time()
  dataRDD \
    .reduceByKey(lambda accu, curr: accu + curr) \
    .sortBy(lambda item: (-item[1], item[0])) \
    .take(10)
  res['exe_time'] = time.time() - start_time
  return res

sc = SparkContext('local[*]', 'task2')
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
n_partition = -1 if len(sys.argv) == 3 else int(sys.argv[3])

dataRDD = sc.textFile(input_file_path) \
  .map(lambda item: json.loads(item)) \
  .map(lambda x: (x['business_id'], 1))

result = dict()

# Default partition function
result['default'] = get_top10_business_analysis(dataRDD)

# Customized partition function
if n_partition != -1:
  dataRDD = dataRDD.partitionBy(n_partition)
result['customized'] = get_top10_business_analysis(dataRDD)

with open(output_file_path, 'w') as output_file:
  output_file.write(json.dumps(result, indent=2))
output_file.close()