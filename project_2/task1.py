from pyspark import SparkContext
import sys
import json

sc = SparkContext('local[*]', 'task1')
case_number = sys.argv[1]
support = sys.argv[2]
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]

print("==============================")
print()
print('case_number:', case_number)
print('support:', support)
print('input_file_path:', input_file_path)
print('output_file_path:', output_file_path)
print()
print("==============================")

