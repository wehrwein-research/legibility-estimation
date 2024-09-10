import sys, os, csv, math
from pathlib import Path
import warnings
warnings.simplefilter("ignore") # Change the filter in this process
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
import numpy as np
import random
import torch 
import argparse

PROJ_ROOT = str(Path(*Path.cwd().parts[:Path().cwd().parts.index('legibility-estimation')+1]))
sys.path.append(PROJ_ROOT + '/MODEL/GMM')
sys.path.append(PROJ_ROOT + '/data')
sys.path.append(PROJ_ROOT + "/MODEL/bordercut")

import border_utils as bu
from gmmy_bear import create_tri_mask
from model import SiameseDeciderOld, BinaryFinetuner, get_binary_train_and_val_loader, BinaryModel

def import_modules():
    import random
    import numpy as np
    import torch 
    sys.path.append(PROJ_ROOT + "/MODEL/bordercut")
    from mix_binary.model import SiameseDeciderOld, BinaryFinetuner, get_binary_train_and_val_loader, BinaryModel

def load_model(weight_path, og_weight_path=PROJ_ROOT + "/MODEL/bordercut/weights/bordercut_backbone.ckpt", 
        concat=True, finetuned=False, test_only_resnet=False, model_type="vit"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if finetuned:
        trained_model = SiameseDeciderOld.load_from_checkpoint(og_weight_path)
        model = BinaryFinetuner.load_from_checkpoint(weight_path, concat=concat, trained_model=trained_model, map_location=device, 
            test_only_resnet=test_only_resnet)
    else:
        model = BinaryModel.load_from_checkpoint(weight_path, map_location=device, backbone=model_type)
    model = model.to(device)
    model.eval()
    return model

def test_binary_model(train_path, val_path, prefix=PROJ_ROOT+'/bing_maps/global-rectified-point4/', weight_path=None, model=None, concat=True, doImport=True, val_score_file=None):
    if (weight_path is None) and (model is None):
        print("Error: Weight path and model both None")
        exit()
    elif doImport:
        import_modules()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model is None:
        model = load_model(weight_path, concat=concat)

    train, val = get_binary_train_and_val_loader(train_path, prefix, False, val_path, batch_size=1)
    def score_function(x):
        x = bu.imread(x)
        x = train.dataset.prep_img(x).unsqueeze(0).to(device)
        y_hat = model(x).cpu().squeeze().detach().tolist()
        return y_hat

    # f: file to write label and pred to (must be opened before and closed after function)    
    def compute_preds(dataloader, f=None):
        correct = 0
        total = 0
        for feature, label in iter(dataloader):
            #pred = model.forward(feature.to(device)).cpu().squeeze().detach()
            pred = model(feature.to(device)).cpu().squeeze().detach()
            if round(float(pred)) == round(float(label)):
                correct += 1
            total += 1
            if f is not None:
                f.write(str(float(label)) + "," + str(float(pred))+ '\n')
        return correct/total

    id_to_file = lambda idee: prefix + idee[:7] + "/" + idee + "/" + idee + ".jpeg"

    train_acc = compute_preds(train)
    if val_score_file is not None:
        f = open(val_score_file, 'w')
        f.write("legible-agreement,score\n")
        val_acc = compute_preds(val, f)
        f.close()
    else:
        val_acc = compute_preds(val)

    return [train_acc, val_acc]

    gts = []
    choices = []
    with open(file, "r") as f:
        for count, line in enumerate(f):
            # check if this file has a csv header
            if count == 0:
                if "tile" in line:
                    continue
            tile, gt = line[:-1].split(",")[:2]
            output = score_function(id_to_file(tile))
            gts.append(gt)
            #print(str(output) + ',' + gt)
            if output < 0.5:
                choices.append("0")
            else:
                choices.append("1")
    total = 0
    correct = 0
    for i, choice in enumerate(choices):
        if gts[i] == choice:
            correct += 1
        total += 1
    return float(correct)/float(total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrape_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["bordercut", "vit", "resnet"], required=True)
    parser.add_argument("--model_path", type=str, default=None)

    args = parser.parse_args()

    files = []
    file1 = PROJ_ROOT + "/MODEL/bordercut/data/train.csv"
    file2 = PROJ_ROOT + "/MODEL/bordercut/data/val.csv"
    file3 = PROJ_ROOT + "/MODEL/bordercut/data/test.csv"

    files.append(file1)
    files.append(file2)
    files.append(file3)

    prefix = args.scrape_dir
    if prefix[-1] != "/":
        prefix += "/"

    weight_paths = []
    if args.model_type == "bordercut":
        concat = True
        finetuned = True
        if args.model_path is None: 
            weight_paths.append(f'{PROJ_ROOT}/MODEL/bordercut/weights/finetuned_model.ckpt')
        else:
            weight_paths.append(args.model_path)
    elif args.model_type == "resnet":
        concat = False
        finetuned = False
        if args.model_path is None: 
            weight_paths.append(f'{PROJ_ROOT}/MODEL/bordercut/weights/resnet.ckpt')
        else:
            weight_paths.append(args.model_path)
    else:
        concat = False
        finetuned = False
        if args.model_path is None: 
            weight_paths.append(f'{PROJ_ROOT}/MODEL/bordercut/weights/vit.ckpt')
        else:
            weight_paths.append(args.model_path)
    
    with open("accuracy.csv", "a") as f:
        writer = csv.writer(f)
        for weight_path in weight_paths:
            train_acc = 0.0
            val_acc = 0.0
            test_acc = 0.0

            run_name = weight_path.split('/')[-1].split(".")[0]
            val_score_file = PROJ_ROOT + "/MODEL/binary/outputs/" + run_name + "-val.csv"
            test_score_file = PROJ_ROOT + "/MODEL/binary/outputs/" + run_name + "-test.csv"

            model = load_model(weight_path, concat=concat, finetuned=finetuned, model_type=args.model_type)

            train_acc, val_acc = test_binary_model(file1, file2, prefix=prefix, model=model, doImport=False, concat=concat, 
                val_score_file=val_score_file)
            train_acc, test_acc = test_binary_model(file1, file3, prefix=prefix, model=model, doImport=False, concat=concat, 
                val_score_file=test_score_file)
            
            writer.writerow([run_name, train_acc, val_acc, test_acc])
