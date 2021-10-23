with open("publicdata/pure_jaccard_similarity.csv") as in_file:
    answer = in_file.read().splitlines(True)[1:]
answer_set = set()
for line in answer:
    row = line.split(',')
    answer_set.add((row[0], row[1]))
with open("outputs/task1_result.csv") as in_file:
    estimate = in_file.read().splitlines(True)[1:]
estimate_set = set()
for line in estimate:
    row = line.split(',')
    estimate_set.add((row[0], row[1]))
print("Precision:")
print(len(answer_set.intersection(estimate_set))/len(estimate_set))
print("Recall:")
print(len(answer_set.intersection(estimate_set))/len(answer_set))
