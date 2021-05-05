import numpy as np

#####################################################################
#   SUBMODULAR FUNCTIONS FOR VIDEO SUMMARIZATION BASED ON PAPER
#
#   1. INTERESTINGNESS : Weighted Coverage function
#   2. REPRESENTATIVENESS : Submodular objective based on K-mediod obj
#                          [Represent a segment using average of global features]
#   3. UNIFORMITY : Submodular objective based on K-mediod obj
#                          [Represent a segment using mean frame number]
#
#
#   LOSS FUNCTION
#
#   1. Recall Loss
#
#
######################################################################


# INTERESTINGNESS - [OVERLAPPING SEGMENTS]

def interestingness_shell(sumData):
    frame_scores = sumData.getFrameScores()
    segment_size = sumData.seg_size
    return lambda selectedIndices: weighted_coverage(selectedIndices, segment_size, frame_scores)


def weighted_coverage(selected_indices, segment_size, frame_scores):
    unionIndices = np.zeros(len(frame_scores))
    for si in selected_indices:
        for j in range(segment_size):
            if si + j < len(frame_scores):
                unionIndices[si+j] = 1
    coverageScore = np.sum(np.dot(unionIndices, frame_scores.T))
    return coverageScore

# REPRESENTATIVENESS - [global features of segment frames are averaged]


def representativeness_shell(sumData):
    '''
    Representativeness shell Eq. (8)
    :param sumData: DataElement with function getDistances()
    :return: representativeness objective
    '''
    distanceMatrix = sumData.getDistances()
    normalizer = distanceMatrix.mean()
    return lambda selectedIndices: (1 - kmedoid_loss(selectedIndices, distanceMatrix, float(normalizer)))

# UNIFORMITY - [mean frame number of global features of segment are considered]


def uniformity_shell(sumData):
    '''
    Based on representativeness_shell implementation in 'example_objectives.py'
    :input S: DataElement with function getChrDistances()
    :return: uniformity objective
    '''
    distanceMatrix = sumData.getChrDistances()
    normalizer = distanceMatrix.mean()
    return lambda selectedIndices: (1 - kmedoid_loss(selectedIndices, distanceMatrix, float(normalizer)))


def kmedoid_loss(selectedIndices, distanceMatrix, norm):
    '''
    :param selectedIndices: selected indices
    :param distanceMatrix: distance matrix
    :param norm: normalizer. defined the distance to the phantom element
    :return: k-medoid loss
    '''
    if len(selectedIndices) > 0:
        min_dist = distanceMatrix[:, selectedIndices].min(axis=1)
        min_dist[min_dist > norm] = norm
        return min_dist.mean()/norm
    else:
        return 1

# RECALL LOSS - [Count of candidate summary y not represented in ground truth]


def recall_loss(sumData, selectedIndices):
    '''
    :param sumData: A DataElement
    :param selectedIndices: a list of selected indices
    :return: the loss (in  [0; 1])
    '''

    budget = sumData.budget
    rl1 = (len(selectedIndices) - len(set(sumData.y_gt).intersection(selectedIndices)))
    return rl1/budget

#########################################################################################
# -------------------------------- OTHER SHELLS ----------------------------------------
#########################################################################################


def random_shell(sumData):
    '''
     Random shell (to check noise sensitivity). Assigns each element in S.Y a random value.
     The score of a solution is the sum over the random values of this solution
    :param sumData: DataElement
    :return: random objective
    '''

    randScores = np.random.rand(len(sumData.Y))
    return lambda selectedIndices: np.sum(randScores[selectedIndices]) / float(sumData.budget)


def x_coord_shell(sumData):
    return lambda selectedIndices: np.sum(sumData.x[np.array(selectedIndices), 0]) / (sumData.budget * sumData.x[:, 0].max())


def earliness_shell(sumData):
    '''
    :param sumData: DataElement
    :return: earliness objective
    '''
    return lambda selectedIndices: (np.max(sumData.Y) * len(selectedIndices) - np.sum(sumData.Y[selectedIndices])) / float(sumData.budget * np.max(sumData.Y))

#########################################################################################
# -------------------------------- OTHER LOSS FUNCTIONS ---------------------------------
#########################################################################################

def intersect_complement_loss(sumData, selectedIndices):
    '''
    :param sumData: A DataElement
    :param selectedIndices: a list of selected indices
    :return: the loss (in  [0; 1])
    '''

    #set intersection is much faster that numpy intersect1d
    return (len(selectedIndices) - len(set(sumData.y_gt).intersection(selectedIndices))) / float(len(sumData.y_gt))
