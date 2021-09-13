from pyspark import SparkContext

sc = SparkContext('local[*]', 'wordCount')

input_file_path = './text.txt'
textRDD = sc.textFile(input_file_path)

counts = textRDD \
    .flatMap(lambda line: line.split(' ')) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .collect()

for each_word in counts:
    print(each_word)
