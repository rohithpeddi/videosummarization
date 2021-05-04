from seq_dpp_linear import SeqDppLinear
import numpy as np

rng_seed = 100
print("seqDPP-linear on OVP")
seqDPP = SeqDppLinear(dataset='OVP', rng_seed=rng_seed)
print("-----------------------------------------------------------------")
print("Start training!!")

# W, alpha, fval = seqDPP.train_dpp_linear_MLE()
#
# W = np.asarray(W)
#
# np.savetxt('w.csv', W, delimiter=',')


print("-----------------------------------------------------------------")
print("Start generating summaries!!")
W = np.genfromtxt('w.csv', delimiter=',')
alpha = 1e-6
videos_te, Ls = seqDPP.test_dpp_inside([seqDPP.videos[i] for i in seqDPP.inds_te], W, alpha, inf_type='exact')

print("-----------------------------------------------------------------")
print("Start evaluation!!")
dataset = 'OVP'
approach_name = 'Linear_' + dataset
result = seqDPP.seqDPP_evaluate(videos_te, seqDPP.inds_te, 1, approach_name)
print('F-score, Recall, Precision: ')
print(result)
