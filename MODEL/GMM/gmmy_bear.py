import sys
from pathlib import Path, PosixPath
from itertools import permutations, combinations

PROJ_ROOT = Path(*Path.cwd().parts[:Path().cwd().parts.index('legibility-estimation')+1])
sys.path.append(str(PROJ_ROOT) + '/data')
import border_utils as bu

import numpy as np
import matplotlib.pyplot as plt

def create_tri_mask(tile, old_dims, new_dims, color=2, thick=5, show=False, one_hot=False):
    lines = None
    if type(tile) in [PosixPath, str]:
        tile = str(Path(tile).with_suffix('.npy'))
        try:
            lines = np.load(tile)
            if lines.shape[0] < 1:
                return None
        except Exception as e:
            print(e)
            return
    else:
        lines = tile
    lines = bu.resize_segments(lines, old_dims, new_dims)
    mask = bu.mask_from_segments(lines, dims=new_dims, draw_lines=True,
                                 color=color, thick=thick)
    # side1 = 0 (red), side2 = 1 (green), side3 = 2 (blue)
    tri_mask = bu.bfs(mask)

    if (np.sum(tri_mask == 0) < 10 or
            np.sum(tri_mask == 1) < 10 or
            np.sum(tri_mask == color) < 10):
        # Not filled
        tri_mask = None

    if show:
        tri_img = np.stack([
            (tri_mask == 0)*255,
            (tri_mask == 1)*255,
            (tri_mask == 2)*255], axis=2).astype(np.uint8)
        print('Ground Truth Mask:')
        print('side0 = red, side1 = green, side2 = blue (border)')
        plt.imshow(tri_img); plt.show()
        
    if one_hot:
        vec_zero = (tri_mask == 0)
        vec_one = (tri_mask == 1)
        if color > 1:
            return np.stack([
                vec_zero,
                vec_one,
                tri_mask == color], axis=2).astype(np.uint8)
        else:
             return np.stack([
                vec_zero,
                vec_one]).astype(np.uint8)
    return tri_mask
