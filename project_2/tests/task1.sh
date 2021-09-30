case_number=$1
support=$2

# Default dataset
input_file_path="./publicdata/small1.csv"
# input_file_path="./publicdata/small2.csv"

output_file_path="./outputs/task1_result.json"

rm -rf $output_filepath
spark-submit --executor-memory 4G --driver-memory 4G task1.py $case_number $support $input_file_path $output_file_path
open $output_filepath                       
