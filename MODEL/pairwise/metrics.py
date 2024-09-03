import pandas as pd
import math

# For cases where our expirement is missing a tile,
# we remove that tile from ground truth
def make_rankings_same_size(r1, r2):
    if len(r1) != len(r2):
        print(len(r1), len(r2))
    r2_set = set(r2)
    r1 = [x for x in r1 if x in r2_set]
    return r1, r2

def kendall_distance(gt_rank, new_rank):
    if len(gt_rank) != len(new_rank):
        gt_rank, new_rank = make_rankings_same_size(gt_rank, new_rank)
    index_map = {rank:i for i, rank in enumerate(new_rank)}
    score = 0
    
    for i, rank in enumerate(gt_rank):
        for j, rank2 in enumerate(gt_rank):
            sigma_i = index_map[rank]
            sigma_j = index_map[rank2]
            
            if i < j and sigma_i > sigma_j:
                score += 1
    return score

def tau(gt_rank, new_rank):
    if len(gt_rank) != len(new_rank):
        gt_rank, new_rank = make_rankings_same_size(gt_rank, new_rank)
    distance = kendall_distance(gt_rank, new_rank)
    return (1)-((4*distance)/(len(gt_rank)*(len(gt_rank)-1)))
                
            
def footrule(gt_rank, new_rank):
    if len(gt_rank) != len(new_rank):
        gt_rank, new_rank = make_rankings_same_size(gt_rank, new_rank)
    score=0  
    for i, rank in enumerate(gt_rank):
        j = new_rank.index(rank)
        score += abs(i-j)
    return score/len(gt_rank)

def binning(gt_df, new_df, num_bins=5, custom_bins=None):   
    values_gt = gt_df["rank"].values
    if custom_bins is None:
        bins = pd.cut(values_gt, num_bins)
    else:
        bins = pd.cut(values_gt, custom_bins)

    gt_df["bin"] = bins
    l = []
    for i in range(new_df.shape[0]):
        row = gt_df[gt_df["name"] == new_df.iloc[i]["name"]]
        name = row["name"].values[0]
        model = new_df.iloc[i]["rank"]
        gt = row["rank"].values[0]
        correct_bin = model in row["bin"].values
        l.append([name, model, gt, correct_bin])
    df = pd.DataFrame(l, columns = ["name", "model", "gt", "correct_bin"])
    num = sum(df["correct_bin"] == True)
    return num / df.shape[0]

# Calculates the spearman/pearson correlation based on type of input.
# If gt and new are lists of the ranks, this will calculate the spearman correlation.
# If gt and new are lists of the raw scores, this will calculate the pearson correlation.
def pearson(gt, new):
    if len(gt) != len(new):
        gt, new = make_rankings_same_size(gt, new)
    xy = []
    x2 = []
    y2 = []

    for i, ranks in enumerate(zip(gt, new)):
        xy.append(ranks[0]*ranks[1])
        x2.append(ranks[0]*ranks[0])
        y2.append(ranks[1]*ranks[1])

    n = len(gt)
    numerator = (n * sum(xy)) - (sum(gt)*sum(new))
    d1 = ((n * sum(x2)) - pow(sum(gt), 2))
    d2 = ((n * sum(y2)) - pow(sum(new), 2))
    denominator = math.sqrt(d1*d2)
    return numerator / denominator
