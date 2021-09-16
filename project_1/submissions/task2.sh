output_filepath="./task2_result.json"
n_partition=$1

rm -rf $output_filepath
/home/local/spark/latest/bin/spark-submit --executor-memory 4G --driver-memory 4G task1.py $ASNLIB/publicdata/test_review.json $output_filepath $n_partition
