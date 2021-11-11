filter_threshold=7
input_file_path="./publicdata/ub_sample_data.csv"
community_output_file_path="./outputs/task1_result.txt"

rm -rf $community_output_file_path
spark-submit --executor-memory 4G --driver-memory 4G --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 task1.py $filter_threshold $input_file_path $community_output_file_path