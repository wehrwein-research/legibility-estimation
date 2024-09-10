import sys, math, csv
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import argparse
sys.path.append('../GMM')
sys.path.append('../../data')
sys.path.append('..')
import border_utils as bu
PROJ_ROOT = str(Path(*Path.cwd().parts[:Path().cwd().parts.index('legibility-estimation')+1]))

def run_deep_cluster_entropy(layer=1, whole=True):
    from methods import deep_cluster_entropy
    if whole == True:
        model_name = f'deep-cluster-entropy-kl-WHOLE-layer{str(layer)}'
    else:
        model_name = f'deep-cluster-entropy-kl-layer{str(layer)}'

    run(deep_cluster_entropy(layer=layer, whole=whole), model_name)

def run_pixel_cluster_entropy():
    from methods import pixel_cluster_entropy
    run(pixel_cluster_entropy(), "pixel-cluster-entropy")

def run_pixel_similarity(mode=0, metric='cosine'):
    from methods import pixel_similarity
    run(pixel_similarity(mode, metric), 'pixel-' + str(mode) + '-' + metric)

def run_deep_feature_similarity(layer=1, mode=0, metric='cosine'):
    from methods import deep_feature_similarity
    run(deep_feature_similarity(layer=layer, mode=mode, metric=metric), 'deep-' + str(mode) + '-' + metric + '-layer'+str(layer))

def run_binary_siamese(weight_path, img_file, finetuned=True, model_type="vit"):
    import torch
    sys.path.append("../bordercut")
    from bordercut.model import SiameseDeciderOld, BinaryFinetuner, BinaryModel
    from bordercut.model import get_binary_train_and_val_loader as get_binary_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if finetuned:
        trained_model = SiameseDeciderOld.load_from_checkpoint(PROJ_ROOT + "/MODEL/bordercut/weights/bordercut_backbone.ckpt")
        model = BinaryFinetuner.load_from_checkpoint(weight_path, trained_model=trained_model, map_location=device)
    else:
        model = BinaryModel.load_from_checkpoint(weight_path, map_location=device, backbone=model_type)
    model = model.to(device)
    model.eval()

    train, val = get_binary_loader(img_file, PREFIX, augment=False, shuffle_train=False)
    def score_function(x):
        x = bu.imread(x)
        x = train.dataset.prep_img(x, augment=False).unsqueeze(0).to(device)
        y_hat = model(x).cpu().squeeze().detach().tolist()
        return y_hat
    run(score_function, weight_path.split("/")[-1].split(".")[0])

def run(score_function, model_name):
    print("Running " + model_name)
    id_to_file = lambda idee: PREFIX + idee[:7] + "/" + idee + "/" + idee + ".jpeg"
    tiles = []
    model_scores = []
    remove_indexes = []
    with open(TILE_FILE, "r") as tf:
        for count, tile in enumerate(tf):
            tiles.append(tile[:-1])
            try:
                score = score_function(id_to_file(tile[:-1]))
                if math.isnan(score):
                    print("NaN score for ", tile[:-1])
                    remove_indexes.append(count)
                else:
                    if PRINT_SCORE:
                        print(tile[:-1], score)
                    model_scores.append(score)
            except Exception as e:
                print("Error for ", tile[:-1])
                print(e)
                remove_indexes.append(count)
    gt = []
    with open(ANNOTATIONS_FILE, "r") as af:
        for count, line in enumerate(af):
            gt.append(int(line.split(",")[1]))

    for i in reversed(remove_indexes):
        gt.pop(i)

    with open(SAVE_FILE, "a", newline='') as save:
        writer = csv.writer(save)
        writer.writerow([model_name,roc_auc_score(gt, model_scores)])
    print(model_name,roc_auc_score(gt, model_scores))
    df = pd.DataFrame()
    df['gt'] = gt
    df['model_scores'] = model_scores
    df.to_csv(RESULTS_DIR + model_name + ".csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--scrape_dir", type=str, required=True)
    parser.add_argument("-verbose", action="store_true")
    parser.add_argument("-clustering", action="store_true")
    parser.add_argument("-distance", action="store_true")
    parser.add_argument("--model_type", type=str, 
        choices=["bordercut", "resnet", "vit", None], default=None)
    parser.add_argument("--model_path", type=str, default=None)

    args = parser.parse_args()

    # file to save model results
    SAVE_FILE = 'roc_auc.csv'

    # directory of scrape
    PREFIX = args.scrape_dir
    if PREFIX[-1] != "/":
        PREFIX += "/"

    # path to annotations file
    ANNOTATIONS_FILE = PROJ_ROOT + "/annotations/test/gt.csv"

    # path to file containing tile names
    TILE_FILE = PROJ_ROOT + "/annotations/test/tiles.txt"
    
    RESULTS_DIR = "outputs/roc_outputs"

    PRINT_SCORE = args.verbose

    #########################################################
    ### Baseline methods:

    ## Clustering:
    if args.clustering:
        # Pixel
        run_pixel_cluster_entropy()
        # Convolutional Feature Layers
        run_deep_cluster_entropy(layer=1)
        run_deep_cluster_entropy(layer=2)
        run_deep_cluster_entropy(layer=3)
    if args.distance:
        ## Distance
        # Pixel
        run_pixel_similarity()
        # Convolutional Feature Layers
        run_deep_feature_similarity(layer=1)
        run_deep_feature_similarity(layer=2)
        run_deep_feature_similarity(layer=3)

    #########################################################
    if args.model_type == "bordercut":
        if args.model_path is None:
            weight_path = f'{PROJ_ROOT}/MODEL/bordercut/weights/finetuned_model.ckpt'
        else:
            weight_path = args.model_path
        run_binary_siamese(weight_path, ANNOTATIONS_FILE, finetuned=True)
    if args.model_type == "resnet":
        if args.model_path is None:
            weight_path = f'{PROJ_ROOT}/MODEL/bordercut/weights/resnet.ckpt'
        else:
            weight_path = args.model_path
        run_binary_siamese(weight_path, ANNOTATIONS_FILE, finetuned=False, model_type="resnet")
    if args.model_type == "vit":
        if args.model_path is None:
            weight_path = f"{PROJ_ROOT}/MODEL/bordercut/weights/vit.ckpt"
        else:
            weight_path = args.model_path
