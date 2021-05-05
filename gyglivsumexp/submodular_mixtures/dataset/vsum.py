import numpy as np
import scipy.spatial.distance as dist
from ..utils import utilities


class VSum(utilities.DataElement):

    def __init__(self, budget, x, Y, gt, gt_score, seg_size):
        self.x = x
        self.Y = Y
        self.y_gt = gt
        self.budget = budget
        self.dist_v = dist.pdist(x)
        self.gt_score = gt_score
        self.seg_size = seg_size
        img_id = list(range(len(x)))
        fno_arr = np.expand_dims(np.array(img_id), axis=1)
        self.dist_c = dist.pdist(fno_arr, 'sqeuclidean')

    def getCosts(self):
        return np.ones(len(self.Y))

    def getDistances(self):
        d = dist.squareform(self.dist_v)
        return np.multiply(d, d)

    def getChrDistances(self):
        d = dist.squareform(self.dist_c)
        return np.multiply(d, d)

    def getFrameScores(self):
        return self.gt_score
