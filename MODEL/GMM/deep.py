from itertools import permutations, combinations
import sys
from pathlib import Path, PosixPath
from pprint import pprint

import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import models, transforms
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

sys.path.append('../../data')
import border_utils as bu
from gmmy_bear import create_tri_mask

def get_model_1(modelBig):
    return torch.nn.Sequential(
        modelBig.conv1,
        modelBig.bn1,
        modelBig.act1,
        modelBig.maxpool,
        modelBig.layer1
    )
def get_model_2(modelBig):
    return torch.nn.Sequential(
        modelBig.conv1,
        modelBig.bn1,
        modelBig.act1,
        modelBig.maxpool,
        modelBig.layer1,
        modelBig.layer2
    )   

def get_model_3(modelBig):
    return torch.nn.Sequential(
        modelBig.conv1,
        modelBig.bn1,
        modelBig.act1,
        modelBig.maxpool,
        modelBig.layer1,
        modelBig.layer2,
        modelBig.layer3
    )

def prep_model(layer=1):
    modelBig = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=0, global_pool='')
    config = resolve_data_config({}, model=modelBig)
    transform = create_transform(**config)
    model = None
    if layer == 1:
        model = get_model_1(modelBig) 
    elif layer == 2:
        model = get_model_2(modelBig) 
    elif layer == 3:
        model = get_model_3(modelBig) 
    try:
        if torch.cuda.is_available():
            model = model.to('cuda')
    except Exception as e:
        print(e)
    return model, transform


'''
args:
    img: np.array (HxWx3) -> image to run prediction on
    model: pytorch model
    transform: function that prepares image to be run by model (from timm package)
'''
def arr2model(img, model, transform):
    img = transforms.functional.to_pil_image(img).convert('RGB')
    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    with torch.no_grad():
        output = model(input_batch).squeeze()
    
    return output

'''
mode: 0 -> one-to-one max
      1 -> one-to-many max
      2 -> one-to-one max-of-differences
      3 -> one-to-many max-of-differences
metric: 'cosine', 'euclidean' , 'similarity'
        see scipy.spatial.distance.cdist for other options
'''
def compute_feature_distance(deep_features, mask, mode, metric):

    dist_to_sim = lambda x : -1 * (x-1)
    similarity = metric == 'similarity'
    if similarity:
        metric = 'cosine'

    A = deep_features[:, mask==0].T
    B = deep_features[:, mask==1].T
    C = deep_features[:, mask==2].T

    if mode == 0:
        dAB = cdist(A,B, metric).mean()
        dAC = cdist(A,C, metric).mean()
        dBC = cdist(B,C, metric).mean()

        if similarity:
            result = 1 - max(dist_to_sim(dAB), dist_to_sim(dAC), dist_to_sim(dBC))
        else:
            result = max(dAB, dAC, dBC)
    elif mode == 1:
        AB = deep_features[:, mask!=2].T
        AC = deep_features[:, mask!=1].T
        BC = deep_features[:, mask!=0].T

        dA_BC = cdist(A,BC, metric).mean()
        dB_AC = cdist(B,AC, metric).mean()
        dC_AB = cdist(C,AB, metric).mean()
        
        if similarity:
            result = 1 - max(dist_to_sim(dA_BC), dist_to_sim(dB_AC), dist_to_sim(dC_AB))
        else:
            result = max(dA_BC, dB_AC, dC_AB)
    elif mode == 2:
        dAB = cdist(A,B, metric).mean()
        dAC = cdist(A,C, metric).mean()
        dBC = cdist(B,C, metric).mean()

        d1 = abs(dAB - dAC)
        d2 = abs(dAB - dBC)
        d3 = abs(dAC - dBC)

        result = max(d1, d2, d3)

    elif mode == 3:
         AB = deep_features[:, mask!=2].T
         AC = deep_features[:, mask!=1].T
         BC = deep_features[:, mask!=0].T

         dA_BC = cdist(A,BC, metric).mean()
         dB_AC = cdist(B,AC, metric).mean()
         dC_AB = cdist(C,AB, metric).mean()

         d1 = abs(dA_BC - dB_AC)
         d2 = abs(dA_BC - dC_AB)
         d3 = abs(dB_AC - dC_AB)

         result = max(d1, d2, d3)
        
    else:
        result = None

    return result

def feature_comp(feats1,  feats2, dist):
    D, M = feats1.shape
    D, K = feats2.shape
        
    res = np.zeros(M)
    # Vectorize all comparisons from one 
    for m in range(M):
        feat = feats1[:, m, None].repeat(1, K)
        res[m] = dist(feat, feats2).mean()
    return res

def run_func_on_sides(image_file, func, model, transform, mode, metric, vis=False):
    image = bu.imread(image_file)
    features = arr2model(image, model, transform)
    h, w = features.shape[1:]
    mask = create_tri_mask(image_file, image.shape[:-1], (h, w), color=2, 
                           thick=2, show=False, one_hot=False)
    if mask is None:
        raise Exception('no mask for ' + image_file)
    return func(features, mask, mode, metric)

def patch_comp(img_file, get_features_func, compare_func, tile_size=(1280, 1280), crop_size=(256, 256)):
    img = bu.imread(img_file)
    side0s, side1s, borders = bu.get_crops_from_tile(img_file, 
                                                     tile_size=tile_size, 
                                                     crop_size=crop_size)
    crop2tile = lambda x: get_features_func(img[x[0]:x[2], x[1]:x[3], :])
    res = [[], [], []]
    for side0 in side0s:
        a = crop2tile(side0)
        for side1 in side1s:
            b = crop2tile(side1)
            res[0].append(compare_func(a, b))
            for border in borders:
                c = crop2tile(border)
                res[1].append(compare_func(b, c))
                res[2].append(compare_func(a, c))
                
    sides = np.array(res[0]).mean()
    one2border = np.array(res[1]).mean()
    zero2border = np.array(res[2]).mean()
    #print('sides:', sides, 
    #      'one2border:', one2border,
    #     'zero2border:', zero2border)
    # Maximum difference between all the differences
    score = max([abs(a-b) for a, b in combinations(
        [sides, one2border, zero2border], r=2
    )])
    #print('max:', score)
    return score

def distribution_kmeans_preds(img, mask, comp_func, model=lambda x: x, n_clusters=3, vis=False, whole_to_side=False):
    img_file = img
    if type(img) in [str, PosixPath]:
        img = bu.imread(img)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, max_iter=600, random_state=12)

    feats = model(img)
    h, w, c = feats.shape
    X = feats.reshape(((h*w), c)).detach().cpu().numpy()
    fitted = kmeans.fit(X)

    all_preds = fitted.labels_
    cluster_img = all_preds.reshape((h,w))

    if vis:
        bu.imshow(img, cluster_img, mask, size=16)
    scores = {}
    centers = {}

    eps = 1e-8
    all_score = np.array([
        np.clip((all_preds == i).sum() / len(all_preds), eps, 1-eps) for i in range(n_clusters)
    ])
    for c in range(3):
        preds = cluster_img[mask == c]
        n = len(preds)
        scores[c] = np.array([
            np.clip((preds == i).sum() / n, eps, 1-eps) for i in range(n_clusters)
        ])
        
        # compare cluster distances if no func
        if comp_func is None:
            centers[c] = KMeans(n_clusters=1, n_init=50, max_iter=600, random_state=12) \
                        .fit(patches) \
                        .cluster_centers_ \
                        .mean()

    res = None
    if comp_func is not None:
        if whole_to_side:
            res = {f'all->{k}': comp_func(all_score, v) for k, v in scores.items()}
        else:
            res = {f'{k1}->{k2}': comp_func(v1, v2) for (k1, v1), (k2, v2) in combinations(scores.items(), r=2)}
    else:
        res = centers
        
    if vis:
        if comp_func is not None:
            pprint(res)
        else:
            print('centers')
            pprint(centers)
            print('scores')
        pprint(scores)
        print('\n')
    return max([abs(a-b) for a in res.values() for b in res.values()])

def IOU(mask, cluster_img):
    from tensorflow.keras.metrics import MeanIoU
    import hungarian as hu
    num_classes = 3
    IoU = MeanIoU(num_classes=num_classes)
    IoU.update_state(mask,cluster_img)
    #print("Mean_IoU = ", IoU.result().numpy())
    values = np.array(IoU.get_weights()).reshape(num_classes, num_classes)
    #print(values)
    vals = []
    for i in range(num_classes):
        item0 = values[0,i] / (values[0,0] + values[0,1] + values[0,2] + values[2,i] + values[1,i])
        item1 = values[1,i] / (values[1,1] + values[1,0] + values[1,2] + values[0, i] + values [2, i])
        item2 = values[2,i] / (values[2,2] + values[2,1] + values[2,0] + values[1,i] + values [0,i])
        vals.append(item0)
        vals.append(item1)
        vals.append(item2)
    vals = np.array(vals).reshape(num_classes,num_classes)
    return hu.hungarian_matching(vals)
