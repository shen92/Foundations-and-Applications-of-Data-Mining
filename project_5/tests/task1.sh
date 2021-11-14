input_filename="./publicdata/users.txt"
stream_size=100
num_of_asks=30
output_filename="./outputs/task1_result.txt"

rm -rf $output_filename
python task1.py $input_filename $stream_size $num_of_asks $output_filename

