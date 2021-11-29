input_file="./publicdata/hw6_clustering.txt"
n_cluster=10
output_file="./outputs/task_result.txt"

rm -rf $output_file
spark-submit --executor-memory 4G --driver-memory 4G task.py $input_file $n_cluster $output_file

