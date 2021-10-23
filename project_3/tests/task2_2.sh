train_file_name="./publicdata/yelp_train.csv"
test_file_name="./publicdata/yelp_val.csv"
output_file_name="./outputs/task2_2_result.txt"

rm -rf $output_file_path
spark-submit --executor-memory 4G --driver-memory 4G task2_2.py $train_file_name $test_file_name $output_file_name