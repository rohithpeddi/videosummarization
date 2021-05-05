import numpy as np
import random
import functools
import json
from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
import scipy.spatial.distance as dist
from chainer import serializers

from script.chain import vid_enc, vid_enc_vgg19
from submodular_mixtures.utils import utilities
from submodular_mixtures.func import functions
from submodular_mixtures.utils.obj import objectives as obj



print("Initialized model, constructing training examples!!")


weights = []
for it in range(10):
    training_examples = []
    counter = 0



weights = np.array(weights)
fin_weights = np.sum(weights, axis=1)/10
print(fin_weights)










