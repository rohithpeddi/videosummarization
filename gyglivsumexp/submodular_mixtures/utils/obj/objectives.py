import numpy as np


def representativeness_shell(sumData):
    '''
    Representativeness shell Eq. (8)
    :param sumData: DataElement with function getDistances()
    :return: representativeness objective
    '''
    distanceMatrix = sumData.getDistances()
    normalizer = distanceMatrix.mean()
    return lambda selectedIndices: (1 - kmedoid_loss(selectedIndices, distanceMatrix, float(normalizer)))

def uniformity_shell(sumData):
    '''
    Based on representativeness_shell implementation in 'example_objectives.py'
    :input S: DataElement with function getChrDistances()
    :return: uniformity objective
    '''
    distanceMatrix = sumData.getChrDistances()
    normalizer = distanceMatrix.mean()
    return lambda selectedIndices: (1 - kmedoid_loss(selectedIndices, distanceMatrix, float(normalizer)))


def uniformity(S):
    '''
    Based on representativeness_shell implementation in 'example_objectives.py'
    :input S: DataElement with function getChrDistances()
    :return: uniformity objective
    '''
    tempDMat = S.getChrDistances()
    norm = tempDMat.mean()
    return (lambda X: (1 - kmedoid_loss(X, tempDMat, float(norm))))


def representativeness(S):
    '''
    Based on representativeness_shell implementation in 'example_objectives.py'
    :input S: DataElement with function getDistances()
    :return: representativeness objective
    '''
    tempDMat = S.getDistances()
    norm = tempDMat.mean()
    return (lambda X: (1 - kmedoid_loss(X, tempDMat, float(norm))))

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


def intersect_complement_loss(sumData, selectedIndices):
    '''
    :param sumData: A DataElement
    :param selectedIndices: a list of selected indices
    :return: the loss (in  [0; 1])
    '''

    #set intersection is much faster that numpy intersect1d
    return (len(selectedIndices) - len(set(sumData.y_gt).intersection(selectedIndices))) / float(len(sumData.y_gt))


def recall_loss(sumData, selectedIndices, budget):
    '''
    :param sumData: A DataElement
    :param selectedIndices: a list of selected indices
    :return: the loss (in  [0; 1])
    '''

    rl1 = (len(selectedIndices) - len(set(sumData.y_gt).intersection(selectedIndices)))
    return rl1/budget

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

