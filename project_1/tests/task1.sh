review_filepath="./publicdata/test_review.json"
# review_filepath="../yelpdata/yelp_academic_dataset_review.json"
output_filepath="./outputs/task1_result.json"

rm -rf $output_filepath
spark-submit --executor-memory 4G --driver-memory 4G task1.py $review_filepath $output_filepath                       
