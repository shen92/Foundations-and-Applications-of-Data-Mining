from pyspark import SparkContext
import sys
import time
import collections
import math
from itertools import combinations

sc = SparkContext('local[*]', 'task2')

case_number = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]

start_time = time.time()
data_RDD = sc.textFile(input_file_path)