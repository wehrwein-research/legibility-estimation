import sys
from pathlib import Path
import pandas as pd
import argparse

sys.path.append('../../GMM')
sys.path.append('../../../data')
sys.path.append('../..')

import border_utils as bu

PROJ_ROOT = str(Path(*Path.cwd().parts[:Path().cwd().parts.index('legibility-estimation')+1]))

def run_deep_cluster_entropy(prefix, file, layer=1, whole=True):
    from methods import deep_cluster_entropy
    run(prefix, file, "deep-cluster-entropy-layer"+str(layer), deep_cluster_entropy(layer=layer))

def run_pixel_cluster_entropy(prefix, file):
    from methods import pixel_cluster_entropy
    run(prefix, file, "pixel-cluster-entropy", pixel_cluster_entropy())

def run_pixel_similarity(prefix, file):
    from methods import pixel_similarity 
    run(prefix, file, "pixel-cosine", pixel_similarity())

def run_deep_feature_similarity(prefix, file, layer=1):
    from methods import deep_feature_similarity
    run(prefix, file, "deep-cosine-layer"+str(layer), deep_feature_similarity(layer=layer))

def run(prefix, file, method_name, score_func):
    global SCORES_DIR, CALC_THRESHOLD
    df = pd.read_csv(file)
    tiles = df['tile'].to_list()
    gts = df['legible?'].to_list()
    scores = []
    for tile in tiles:
        tile_path = prefix + tile[:7] + '/' + tile + '/' + tile + '.jpeg'
        try:
            score = score_func(tile_path)
        except Exception as e:
            print(e)
            score = None 
        scores.append(score)
        print(score)
    df['score'] = scores
    score_file = SCORES_DIR + "/" + method_name + "_scores.csv"
    df.to_csv(score_file, index=False)
    if CALC_THRESHOLD:
        run_threshold(score_file, method_name)
    else:
        global TRAIN_RESULTS
        global TEST_RESULTS
        df = pd.read_csv(TRAIN_RESULTS)
        for i, row in df.iterrows():
            if row['method'] == method_name:
                threshold = row['threshold']
                break 
        correct, total = test_thresholds(score_file, [threshold])
        with open(TEST_RESULTS, 'a') as f:
            f.write(method_name)
            f.write("," + str(correct[0]/total[0]) + "\n")

def test_thresholds(scores_file, thresholds):
    print("Testing threshold")
    df = pd.read_csv(scores_file)
    corrects = []
    totals = []
    for item in thresholds:
        corrects.append(0)
        totals.append(0)

    for count, row in df.iterrows():
        gt = int(row['legible?'])
        score = float(row['score'])
        for i, threshold in enumerate(thresholds):
            totals[i] += 1
            guess = 0 if score < threshold else 1 
            if guess == gt:
                corrects[i] += 1
    return corrects, totals

def run_threshold(scores_file, method_name):
    print("Computing threshold for " + method_name)
    global TRAIN_RESULTS
    df = pd.read_csv(scores_file)
    thresholds = df['score'].to_list()
    corrects, totals = test_thresholds(scores_file, thresholds)
    accuracies = []
    highest_accuracy = 0.0
    best_threshold = 0.0
    for threshold,c,t in zip(thresholds, corrects, totals):
        acc = c/t
        if acc > highest_accuracy:
            highest_accuracy = acc
            best_threshold = threshold
    with open(TRAIN_RESULTS, "a") as f:
        f.write(method_name)
        f.write("," + str(highest_accuracy))
        f.write("," + str(best_threshold) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scrape_dir", type=str, required=True)
    parser.add_argument("-calc_threshold", action="store_true")
    parser.add_argument("-test_threshold", action="store_true")
    
    parser.add_argument("-clustering", action="store_true")
    parser.add_argument("-distance", action="store_true")

    args = parser.parse_args()

    # One and only one option must be selected
    assert args.calc_threshold or args.test_threshold
    assert not (args.calc_threshold and args.test_threshold)
    # at least on option must be selected
    assert args.clustering or args.distance

    global SCORES_DIR
    global CALC_THRESHOLD
    global TRAIN_RESULTS
    global TEST_RESULTS
    CALC_THRESHOLD = args.calc_threshold

    prefix = args.scrape_dir
    if prefix[-1] != "/":
        prefix += "/"

    if CALC_THRESHOLD:
        SCORES_DIR = "train-scores"
        file = PROJ_ROOT + "/annotations/train/gt_df.csv"
    else:
        SCORES_DIR = "test-scores"
        file = PROJ_ROOT + "/annotations/test/gt_df.csv"

    TRAIN_RESULTS = "./training.csv"
    TEST_RESULTS = "./testing.csv"
    
    if args.clustering:
        ## Clustering:
        # Pixel
        run_pixel_cluster_entropy(prefix, file)
        # Convolutional Feature Layers
        run_deep_cluster_entropy(prefix, file, layer=1)
        run_deep_cluster_entropy(prefix, file, layer=2)
        run_deep_cluster_entropy(prefix, file, layer=3)
    if args.distance:
        ## Distance
        # Pixel
        run_pixel_similarity(prefix, file)
        # Convolutional Feature Layers
        run_deep_feature_similarity(prefix, file, layer=1)
        run_deep_feature_similarity(prefix, file, layer=2)
        run_deep_feature_similarity(prefix, file, layer=3)

if __name__ == "__main__":
    main()
