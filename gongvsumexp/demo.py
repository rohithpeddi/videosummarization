from seq_dpp_linear import SeqDppLinear
import numpy as np

# rng_seed = 100
# F1s = []
# pre = []
# rec = []
# for i in range(100):
#     rng_seed = rng_seed + np.random.randint(10)
#     print("-----------------------------------------------------------------")
#     print("-----------------------ITERATION + str(i)" + "--------------------")
#     print("seqDPP-linear on OVP")
#     seqDPP = SeqDppLinear(dataset='OVP', rng_seed=rng_seed)
#     print("-----------------------------------------------------------------")
#     print("Start training!!")
#
#     # W, alpha, fval = seqDPP.train_dpp_linear_MLE()
#     #
#     # W = np.asarray(W)
#     #
#     # np.savetxt('w.csv', W, delimiter=',')
#
#
#     print("-----------------------------------------------------------------")
#     print("Start generating summaries!!")
#     W = np.genfromtxt('w.csv', delimiter=',')
#     alpha = 1e-6
#     videos_te, Ls = seqDPP.test_dpp_inside([seqDPP.videos[i] for i in seqDPP.inds_te], W, alpha, inf_type='exact')
#
#     print("-----------------------------------------------------------------")
#     print("Start evaluation!!")
#     dataset = 'OVP'
#     approach_name = 'Linear_' + dataset
#     F1, precision, recall = seqDPP.seqDPP_evaluate(videos_te, seqDPP.inds_te, 1, approach_name)
#     F1s.append(F1)
#     pre.append(precision)
#     rec.append(recall)
#     print('F-score, Recall, Precision: ')
#     print(F1, precision, recall)
#
#
# Avg_F1 = np.sum(np.array(F1s))/100
# Avg_precision = np.sum(np.array(pre))/100
# Avg_recall = np.sum(np.array(rec))/100
#
# print(Avg_F1, Avg_precision, Avg_recall)

# rng_seed = 200
# seqDPP = SeqDppLinear(dataset='Youtube', rng_seed=rng_seed)
# print("-----------------------------------------------------------------")
# print("Start training!!")
#
# W, alpha, fval = seqDPP.train_dpp_linear_MLE()
# W = np.asarray(W)
# np.savetxt('w_you.csv', W, delimiter=',')
# print(str(alpha))

F1s = []
pre = []
rec = []
for i in range(100):
    print("-----------------------------------------------------------------")
    print("-----------------------ITERATION + str(i)" + "--------------------")
    print("seqDPP-linear on Youtube")
    seqDPP = SeqDppLinear(dataset='Youtube', rng_seed=rng_seed)

    print("-----------------------------------------------------------------")
    print("Start generating summaries!!")
    W = np.genfromtxt('w.csv', delimiter=',')
    alpha = 1e-6
    videos_te, Ls = seqDPP.test_dpp_inside([seqDPP.videos[i] for i in seqDPP.inds_te], W, alpha, inf_type='exact')

    print("-----------------------------------------------------------------")
    print("Start evaluation!!")
    dataset = 'OVP'
    approach_name = 'Linear_' + dataset
    F1, precision, recall = seqDPP.seqDPP_evaluate(videos_te, seqDPP.inds_te, 1, approach_name)
    F1s.append(F1)
    pre.append(precision)
    rec.append(recall)
    print('F-score, Recall, Precision: ')
    print(F1, precision, recall)

    rng_seed = rng_seed + np.random.randint(10)

print("STATISTICS: ")
print("F1 : Mean " + str(np.array(F1s).mean() )+ ", Variance : " + str(np.array(F1s).var()))
print("Precision : Mean " + str(np.array(pre).mean() )+ ", Variance : " + str(np.array(pre).var()))
print("Recall : Mean " + str(np.array(rec).mean()) + ", Variance : " + str(np.array(rec).var()))