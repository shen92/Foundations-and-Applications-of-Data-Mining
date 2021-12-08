folder_path="./publicdata/"
test_file_name="./publicdata/yelp_val.csv"
output_file_name="./outputs/competition_result.csv"

rm -rf $output_file_name
spark-submit --executor-memory 4G --driver-memory 4G competition.py $folder_path $test_file_name $output_file_name