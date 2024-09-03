import sys, os, csv, math
import warnings
warnings.simplefilter('ignore')

sys.path.append('GMM')
sys.path.append('../data')

import border_utils as bu

def deep_cluster_entropy(layer=1, whole=True):
    from deep import prep_model, arr2model, distribution_kmeans_preds
    from gmmy_bear import create_tri_mask
    from scipy.special import kl_div
    
    kl = lambda x, y: kl_div(x, y).mean()
    model, transform = prep_model(layer=layer)
    m = lambda x: arr2model(x, model, transform).permute((1,2,0))
    
    # Sets mask properties based on layer
    i = 56
    if layer == 2:
        i = 28
    elif layer == 3:
        i = 14
            
    def score_function(img):
        h, w, c = bu.imread(img).shape
        deep_tri_mask = create_tri_mask(img, (h,w), (i,i), color=2, thick=2, show=False,one_hot=False)
        return distribution_kmeans_preds(img, deep_tri_mask, kl, n_clusters=3, model=m, vis=False, whole_to_side=whole)
    
    return score_function

def pixel_cluster_entropy():
    from deep import prep_model, arr2model, distribution_kmeans_preds
    from gmmy_bear import create_tri_mask
    from scipy.special import kl_div
    
    kl = lambda x, y: kl_div(x, y).mean()
    model, transform = prep_model()
    m = lambda x: arr2model(x, lambda x: x, transform).permute((1,2,0))
    def score_function(img):
        h, w, c = bu.imread(img).shape
        deep_tri_mask = create_tri_mask(img, (h,w), (224,224), color=2, thick=5, show=False)
        return distribution_kmeans_preds(img, deep_tri_mask, kl, n_clusters=3, model=m, vis=False)
    return score_function

def pixel_similarity(mode=0, metric='cosine'):
    from deep import run_func_on_sides, prep_model, compute_feature_distance
    
    model1, transform = prep_model()
    m = lambda x: x
    score_function = lambda img: float(run_func_on_sides(img, compute_feature_distance, m, transform, mode, metric, vis=False))
    return score_function

def deep_feature_similarity(layer=1, mode=0, metric='cosine'):
    from deep import run_func_on_sides, prep_model, compute_feature_distance
    
    model, transform = prep_model(layer=layer)
    score_function = lambda img: float(run_func_on_sides(img, compute_feature_distance, model, transform, mode, metric, vis=False))
    return score_function
