filter_threshold=7
input_file_path="./publicdata/ub_sample_data.csv"
betweenness_output_file_path="./outputs/task2_betweenness.txt"
community_output_file_path="./outputs/task2_result.txt"

rm -rf $betweenness_output_file_path
rm -rf $community_output_file_path
spark-submit --executor-memory 4G --driver-memory 4G task2.py $filter_threshold $input_file_path $betweenness_output_file_path $community_output_file_path