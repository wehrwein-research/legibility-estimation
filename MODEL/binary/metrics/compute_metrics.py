import regression.regression_metrics as rm
import pandas as pd
import numpy as np
import sys
import os
import argparse
sys.path.append("../../pairwise")
from metrics import pearson

"""
metrics_list: [[mse's], [rmse's], [mae's], [pearson's]]
"""
def compute_metrics(file_path, metrics_list, args):
    df = pd.read_csv(file_path, dtype={"legible-agreement": np.float64, "score":np.float64})

    targets = df["legible-agreement"]
    preds = df["score"]

    # cosine ranges from [0,2]
    # bordercut ranges from [0,1]
    if args.transfer == "false":
        norm_preds = preds/ 2.0
    else:
        norm_preds = preds

    missing = norm_preds.isna()

    targets = list(targets[missing==False])
    norm_preds = list(norm_preds[missing==False])
    preds = list(preds[missing==False])

    metrics_list[0].append(str(rm.mse(targets, norm_preds)))
    metrics_list[1].append(str(rm.rmse(targets, norm_preds)))
    metrics_list[2].append(str(rm.mae(targets, norm_preds)))
    metrics_list[3].append(str(pearson(targets, preds)))

parser = argparse.ArgumentParser()

parser.add_argument("--score_file", required=True, type=str)
parser.add_argument("-transfer", action="store_true", help="Use when using scores from BorderCut or other transfer model")

args = parser.parse_args()

methods = []
metrics = [[], [], [], []]
path = args.score_file

if os.path.isfile(path):
    methods.append(path.split("/")[-1].split(".")[0])
    compute_metrics(path, metrics, args)
elif os.path.isdir(path):
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            methods.append(filename.split(".")[0])
            compute_metrics(f, metrics, args)

print("mse,rmse,mae,pearson")
for i in range(len(metrics[0])):
    print(methods[i] + "," + metrics[0][i]+"," + 
    metrics[1][i] + "," + metrics[2][i] + "," + metrics[3][i])
