output_filepath_question_a="./outputs/task3_result_a.txt"
output_filepath_question_b="./outputs/task3_result_b.json"

rm -rf $output_filepath_question_a
rm -rf $output_filepath_question_b
/home/local/spark/latest/bin/spark-submit --executor-memory 4G --driver-memory 4G task3.py $ASNLIB/publicdata/test_review.json $ASNLIB/publicdata/business.json $output_filepath_question_a $output_filepath_question_b
