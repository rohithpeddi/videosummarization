from seq_dpp_linear import SeqDppLinear
import numpy as np

rng_seed = 100
print("seqDPP-linear on OVP")
seqDPP = SeqDppLinear(dataset='OVP', rng_seed=rng_seed)
# W, alpha, fval = seqDPP.train_dpp_linear_MLE()
#
# W = np.asarray(W)
# alpha = np.asarray(alpha)
#
# np.savetxt('w.csv', W, delimiter=',')
# np.savetxt('alpha', alpha,  delimiter=',')


print("-----------------------------------------------------------------")
print("Start generating summaries!!")
W = np.genfromtxt('w.csv', delimiter=',')
alpha = 1e-6
videos_te = seqDPP.test_dpp_inside([seqDPP.videos[i] for i in seqDPP.inds_te], W, alpha, inf_type='exact')
