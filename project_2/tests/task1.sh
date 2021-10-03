input_file_path="./publicdata/small2.csv"
output_case_1_path="./outputs/task1_result_case_1.txt"
output_case_2_path="./outputs/task1_result_case_2.txt"

rm -rf $output_case_1_path
rm -rf $output_case_2_path
spark-submit --executor-memory 4G --driver-memory 4G task1.py 1 4 $input_file_path $output_case_1_path
spark-submit --executor-memory 4G --driver-memory 4G task1.py 2 9 $input_file_path $output_case_2_path
   
