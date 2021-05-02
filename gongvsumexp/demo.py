from seq_dpp_linear import SeqDppLinear

rng_seed = 100
print("seqDPP-linear on OVP")
seqDPP = SeqDppLinear(dataset='OVP', rng_seed=rng_seed)
W, alpha, fval = seqDPP.train_dpp_linear_MLE()

print("-----------------------------------------------------------------")
print("Start generating summaries!!")
videos_te = seqDPP.test_dpp_inside(seqDPP.videos[seqDPP.inds_te], W, alpha, inf_type='exact')