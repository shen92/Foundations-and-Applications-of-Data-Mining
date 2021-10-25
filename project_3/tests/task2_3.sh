folder_path="./publicdata/"
test_file_name="./publicdata/yelp_val.csv"
output_file_name="./outputs/task2_3_result.csv"

rm -rf $output_file_name
spark-submit --executor-memory 4G --driver-memory 4G task2_3.py $folder_path $test_file_name $output_file_name

python tests/validation_task2.py $output_file_name