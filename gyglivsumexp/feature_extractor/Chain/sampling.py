import dataset
import functools
import numpy as np
import scipy.spatial.distance as distance
from ...submodular_mixtures.func import functions
from ...submodular_mixtures.utils import utilities
from ...submodular_mixtures.utils.obj import objectives as obj

class VSUM(utilities.DataElement):

    def __init__(self, videoID, model, ds='summe', featType='vgg', seg_l=5):
        # load dataset data
        self.dataset = dataset.SUMMDATA(videoID)

        # budget 15% of orig
        self.budget = int(0.15 * self.dataset.data['length'] / seg_l)
        print ('budget: ', self.budget)

        # embed video segments
        seg_feat = encodeSeg(self.dataset, model, seg_size=seg_l)

        # store segment features
        self.x = seg_feat
        self.Y = np.ones(self.x.shape[0])

        # compute distance between segments
        self.dist_e = distance.squareform(distance.pdist(self.x, 'sqeuclidean'))

        # compute chronological distance
        self.frame, img_id, self.score = self.dataset.sampleFrame()

        fno_arr = np.expand_dims(np.array(img_id), axis=1)
        self.dist_c = distance.pdist(fno_arr, 'sqeuclidean')

    def getCosts(self):
        return np.ones(self.x.shape[0])

    def getRelevance(self):
        return np.multiply(self.rel, self.rel)

    def getChrDistances(self):
        d = distance.squareform(self.dist_c)
        return np.multiply(d, d)

    def getDistances(self):
        return np.multiply(self.dist_e, self.dist_e)

    def summarizeRep(self, seg_l=5, weights=[1.0, 0.0]):

        objectives = [obj.representativeness_shell(self), obj.uniformity_shell(self)]
        selected, score, minoux_bound = functions.leskovec_maximize(self, weights, objectives, budget=self.budget)
        selected.sort()

        frames = []
        gt_score = []
        for i in selected:
            frames.append(self.frame[i:i + seg_l])
            gt_score.append(self.score[i:i + seg_l])

        return selected, frames, gt_score


def encodeSeg(data, model, seg_size=5):
    feat = data.feat
    # feat = torch.from_numpy(feat)

    img, img_id, score = data.sampleFrame()
    segs = [img_id[i:i + seg_size] for i in range(len(img_id) - seg_size + 1)]
    segs = functools.reduce(lambda x, Y: x + Y, segs)

    x = feat[segs]

    # embedding
    enc_x = model(x)

    return enc_x.data