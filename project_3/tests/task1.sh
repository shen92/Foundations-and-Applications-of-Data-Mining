input_file_name="./publicdata/yelp_train.csv"
output_file_name="./outputs/task1_result.csv"

rm -rf $output_file_name
spark-submit --executor-memory 4G --driver-memory 4G task1.py $input_file_name $output_file_name

python tests/validation_task1.py