input_file_path="./publicdata/ta_feng_all_months_merged.csv"
output_file_path="./outputs/task2_result.txt"
intermediate_file_path="./intermediate.csv"

rm -rf $intermediate_file_path
rm -rf $output_file_path
spark-submit --executor-memory 4G --driver-memory 4G task2.py 20 50 $input_file_path $output_file_path         
