import numpy as np
import sys

output_file_name = sys.argv[1]

with open(output_file_name) as in_file:
    guess = in_file.readlines()[1:]
with open("publicdata/yelp_val.csv") as in_file:
    ans = in_file.readlines()[1:]
res = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
dist_guess = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
dist_ans = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
large_small = {"large": 0, "small": 0}

RMSE = 0
for i in range(len(guess)):
    diff = float(guess[i].split(",")[2]) - float(ans[i].split(",")[2])
    RMSE += diff**2
    if abs(diff) < 1:
        res["<1"] = res["<1"] + 1
    elif 2 > abs(diff) >= 1:
        res["1~2"] = res["1~2"] + 1
    elif 3 > abs(diff) >= 2:
        res["2~3"] = res["2~3"] + 1
    elif 4 > abs(diff) >= 3:
        res["3~4"] = res["3~4"] + 1
    else:
        res["4~5"] = res["4~5"] + 1
RMSE = (RMSE/len(guess))**(1/2)
print("RMSE: "+str(RMSE))
prediction = np.array([float(gg.split(',')[2]) for gg in guess])
print("Prediction mean: " + str(prediction.mean()))
print("Prediction std:" + str(prediction.std()))
ground = np.array([float(gg.split(',')[2]) for gg in ans])
print("Answer mean: "+str(ground.mean()))
print("Answer std: "+str(ground.std()))
