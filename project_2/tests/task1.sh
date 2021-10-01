case_number=$1
support=$2

# Default dataset
input_dev_file_path="./publicdata/small1.csv"
input_test_file_path="./publicdata/small2.csv"

output_file_path="./outputs/task1_result.txt"
output_case_1_path="./outputs/task1_result_case_1.txt"
output_case_2_path="./outputs/task1_result_case_2.txt"

rm -rf $output_file_path
rm -rf $output_case_1_path
rm -rf $output_case_2_path
spark-submit --executor-memory 4G --driver-memory 4G task1.py $case_number $support $input_dev_file_path $output_file_path
spark-submit --executor-memory 4G --driver-memory 4G task1.py 1 4 $input_test_file_path $output_case_1_path
spark-submit --executor-memory 4G --driver-memory 4G task1.py 2 9 $input_test_file_path $output_case_2_path
open $output_file_path           
