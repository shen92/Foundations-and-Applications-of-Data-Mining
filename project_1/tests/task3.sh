review_filepath="./publicdata/test_review.json"
business_filepath="./publicdata/business.json"
# review_filepath="../yelpdata/yelp_academic_dataset_review.json"
# business_filepath="../yelpdata/yelp_academic_dataset_business.json"
output_filepath_question_a="./outputs/task3_result_a.txt"
output_filepath_question_b="./outputs/task3_result_b.json"

rm -rf $output_filepath_question_a
rm -rf $output_filepath_question_b
spark-submit --executor-memory 4G --driver-memory 4G task3.py $review_filepath $business_filepath $output_filepath_question_a $output_filepath_question_b                 
