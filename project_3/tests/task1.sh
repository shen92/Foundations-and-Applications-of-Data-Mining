input_file_name="./publicdata/yelp_train.csv"
# input_file_name="./publicdata/test.csv"
output_file_name="./outputs/task1_result.csv"

rm -rf $output_file_name
spark-submit --executor-memory 4G --driver-memory 4G task1.py $input_file_name $output_file_name