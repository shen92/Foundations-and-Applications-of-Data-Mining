# Default dataset
review_filepath="./publicdata/test_review.json"

# Yelp dataset
# review_filepath="./yelpdata/yelp_academic_dataset_review.json"

output_filepath="./outputs/task2_result.json"
n_partition=$1

rm -rf $output_filepath
spark-submit --executor-memory 4G --driver-memory 4G task2.py $review_filepath $output_filepath $n_partition
open $output_filepath                       
