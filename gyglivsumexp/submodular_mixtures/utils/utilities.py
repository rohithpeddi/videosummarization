import numpy as np
import time

class DataElement:
    '''
    Defines a DataElement.
    For inference, this needs the function getCosts(), and a set Y (candidate elements).
    '''

    def __init__(self):
        Y = []

    def getCosts(self):
        raise NotImplementedError

    def __str__(self):
        return 'DataElement'

class SGDparams:
    '''
        Class for the parameters of stochastic gradient descent used for learnSubmodularMixture
    '''

    def __init__(self, **kwargs):
        self.momentum = 0.0  #: defines the momentum used. Default: 0.0
        self.use_l1_projection = False  #: project the weights into an l_1 ball (leads to sparser solutions). Default: False
        self.use_ada_grad = False  #: use adaptive gradient [6]? Default: False
        self.max_iter = 10  #: number of passes throught the dataset (3-10 should do). Default: 10
        self.norm_objective_scores = False  #: normalize the objective scores to sum to one. This improved the learnt weights and can be considered to be the equivalent to l1 normalization of feature points in a standard SVM
        self.learn_lambda = None  #: learning rate. Default: Estimated using [1]
        self.nu = lambda t, T: 1.0 / np.sqrt(
            t + 1)  #: Function nu(t,T) to compute nu for each iteration, given the current iteration t and the maximal number of iterations T. Default: 1/sqrt(t+1)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return 'SGDparams\n-----\n%s' % '\n'.join(
            map(lambda x: '%22s:\t%s' % (x, str(self.__dict__[x])), self.__dict__.keys()))

def evalSubFun(subFunVector, y, isGt, w=None):
    results = np.zeros(len(subFunVector))
    for id in range(len(subFunVector)):
        if w is not None and w[id] == 0:
            results[id] = 0
        else:
            results[id] = subFunVector[id](y)
    return results


def zero_loss(d, d2):
    return len(d2) * 0.0001


def instaciateFunctions(submod_fun, s):
    fun_list = []
    name_list = []
    for idx in range(0, len(submod_fun)):
        res = submod_fun[idx](s)
        if type(res) is tuple:
            objective = res[0]
            names = res[1]
        else:
            objective = res
            names = submod_fun[idx].__name__

        if type(objective) is list:
            fun_list.extend(objective)
            if names is not list:
                names = map(lambda x: '%s-%d' % (names, x), np.arange(len(objective)))
            name_list.extend(names)
        else:
            fun_list.append(objective)
            if names is None:
                name_list.append(submod_fun[idx].func_name)
            else:
                name_list.append(names)

    return fun_list, name_list


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te - ts))
        return result

    return timed