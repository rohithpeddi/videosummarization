from seq_dpp_linear import SeqDppLinear

rng_seed = 100
print("seqDPP-linear on OVP")
seqDPP = SeqDppLinear(dataset='OVP', rng_seed=rng_seed)
seqDPP.train_dpp_linear_MLE()
