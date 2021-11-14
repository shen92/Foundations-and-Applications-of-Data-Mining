input_filename="./publicdata/users.txt"
stream_size=300
num_of_asks=30
output_filename="./outputs/task2_result.txt"

rm -rf $output_filename
python task2.py $input_filename $stream_size $num_of_asks $output_filename
# python task2_Flajolet_Martin_Algorithm.py $input_filename $stream_size $num_of_asks $output_filename


