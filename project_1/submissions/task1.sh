output_filepath="./task1_result.json"

rm -rf $output_filepath
/home/local/spark/latest/bin/spark-submit --executor-memory 4G --driver-memory 4G task1.py $ASNLIB/publicdata/test_review.json $output_filepath
