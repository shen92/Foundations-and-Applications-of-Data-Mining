from pyspark import SparkContext
import sys
import json

sc = SparkContext('local[*]', 'task2')
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]