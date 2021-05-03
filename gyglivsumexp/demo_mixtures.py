import numpy as np
import functools
from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
import scipy.spatial.distance as dist
from chainer import serializers

from feature_extractor.Chain import vid_enc, vid_enc_vgg19
from submodular_mixtures.utils import utilities
from submodular_mixtures.func import functions
from submodular_mixtures.utils.obj import objectives as obj

datasetRoot = 'data/summe/'
directoryPath = "summe/GT/"
onlyfiles = [f for f in listdir(directoryPath) if isfile(join(directoryPath, f))]
model = vid_enc.Model()
serializers.load_npz('data/trained_model/model_par', model)

class St(utilities.DataElement):

    def __init__(self, budget, x, Y, y_gt):
        self.budget = budget
        self.x = x
        self.Y = Y
        self.y_gt = y_gt
        self.dist_v = dist.pdist(x)
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

training_examples = []
counter = 0
for f in listdir(directoryPath):
    if isfile(join(directoryPath, f)):
        if counter >= 5:
            break
        counter += 1
        mfile = loadmat(join(directoryPath, f))
        video_id = f[:-4]
        frames, numUsers = mfile['user_score'].shape
        randUser = np.random.randint(numUsers)
        user = randUser
        # for user in range(numUsers):

        y_gt = mfile['user_score'][:, user]
        print("Creating data for " + str(video_id) + ' with user summary ' + str(user))
        features = np.load(datasetRoot + 'feat/vgg/' + video_id + '.npy').astype(np.float32)
        nFrames = mfile['nFrames'].flatten()[0]
        img_id = list(range(nFrames))
        seg_size = 5
        segs = [img_id[i:i + seg_size] for i in range(len(img_id) - seg_size + 1)]
        segs = functools.reduce(lambda x, Y: x + Y, segs)
        x = features[segs]
        enc_x = model(x)
        x = enc_x.data
        features_length = len(x)

        diff = np.abs(len(y_gt) - features_length)
        y_gt = y_gt[diff:]
        Y = np.ones(len(y_gt))

        video_duration = mfile['video_duration'].flatten()[0]
        budget = int(0.15 * video_duration / seg_size)
        S = St(budget, x, Y, y_gt)
        training_examples.append(S)

print("Finished creation of training data")
# Learn the weights. Given that we used the k-medoid results as ground truth, this objective should get all the weight
shells=[obj.representativeness,obj.representativeness]
loss=obj.intersect_complement_loss

# Use AdaGrad and a l-1 semiball projection (leads to sparser solutions, i.e. is more robust to noise)
params=utilities.SGDparams(use_l1_projection=True,max_iter=10,use_ada_grad=True)

print("Started learning mixture weights")
learnt_weights,_ = functions.learnSubmodularMixture(training_examples, shells,loss,params=params)

print("Finished learning mixture weights")








