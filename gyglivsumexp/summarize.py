import numpy as np
import random
import json
import functools
from chainer import serializers
from os import listdir
from scipy.io import loadmat
from os.path import isfile, join
from script.chain import vid_enc
from submodular_mixtures.dataset.vsum import VSum
from submodular_mixtures.utils import utilities
from submodular_mixtures.func import functions
from submodular_mixtures.utils.obj import objectives as obj
from sklearn.metrics import f1_score, precision_score, recall_score


############################################################################################
#   From SUMME dataset we sample few videos for training to find mixture weights
#   We test on other videos and present the summarization results in comparison with user summaries
############################################################################################

seg_size = 5
datasetRoot = './data/summe/'
directoryPath = "./summe/GT/"
model = vid_enc.Model()
serializers.load_npz('./data/trained_model/model_par', model)

# Run summarize for 10 times with different rng_seeds inorder to select different set of train videos every time
# Average the F1, precision, recall statistics produced for 10 rounds


def evaluate(predicted, user_summary):
    F1 = f1_score(user_summary, predicted, average="macro")
    precision = precision_score(user_summary, predicted, average="macro")
    recall = recall_score(user_summary, predicted, average="macro")
    return F1, precision, recall


def train(training_files, max_users=10):
    training_examples = []
    for file_name in training_files:
        video_id = file_name[:-4]
        tr_file_mat = loadmat(join(directoryPath, file_name))
        training_features = np.load(datasetRoot + 'feat/vgg/' + video_id + '.npy').astype(np.float32)
        frames, num_users = tr_file_mat['user_score'].shape

        # We sample 3 users from the set of users and use their selection as ground truth
        random_user_list = random.sample(range(0, num_users), max_users)
        for user in random_user_list:
            print("Creating data for " + str(video_id) + ' with user summary ' + str(user))
            S, _ = create_vsum(training_features, tr_file_mat, user)
            training_examples.append(S)

    print("Finished creation of training data")

    # Following the submodular functions shells and loss function from paper we get
    params = utilities.SGDparams(use_l1_projection=False, max_iter=10, use_ada_grad=True)

    print("Started learning mixture weights:  ")
    learnt_weights, _ = functions.learnSubmodularMixture(training_examples, shells, loss, params=params)
    print("Finished learning mixture weights")
    return learnt_weights


def predict(test_files, weights, max_users=15):
    predicted = []
    ground_truth = []
    F1_list = []
    precision_list = []
    recall_list = []
    for file_name in test_files:
        video_id = file_name[:-4]
        test_file_mat = loadmat(join(directoryPath, file_name))
        test_features = np.load(datasetRoot + 'feat/vgg/' + video_id + '.npy').astype(np.float32)
        frames, num_users = test_file_mat['user_score'].shape

        randUser = random.sample(range(0, num_users), 1)
        print("Creating data for " + str(video_id) + ' with user summary ' + str(randUser))
        S, y_gt = create_vsum(test_features, test_file_mat, randUser)
        objective_funcs = [obj.representativeness_shell(S), obj.uniformity_shell(S), obj.interestingness_shell(S)]
        selected, score, minoux_bound = functions.leskovec_maximize(S, weights, objective_funcs, budget=S.budget)
        selected = np.sort(selected)

        predicted_labels = np.zeros(len(S.Y))
        for idx in list(selected):
            for j in range(seg_size):
                # for j in range(1):
                if idx + j < len(S.Y):
                    predicted_labels[idx + j] = 1

        random_user_list = random.sample(range(0, num_users), max_users)
        for user in random_user_list:
            # Use the prediction from this user and ground truth summaries from all other user summaries
            S, y_gt = create_vsum(test_features, test_file_mat, user)

            ground_truth_labels = np.zeros(len(S.Y))
            for idx in list(y_gt):
                ground_truth_labels[idx] = 1

            F1 = f1_score(ground_truth_labels, predicted_labels, average="macro")
            precision = precision_score(ground_truth_labels, predicted_labels, average="macro")
            recall = recall_score(ground_truth_labels, predicted_labels, average="macro")

            print("USER: " + str(user) + ', F1 : ' + str(F1) + ', recall : ' + str(recall))

            F1_list.append(F1)
            precision_list.append(precision)
            recall_list.append(recall)

            # predicted = np.concatenate((predicted, predicted_labels), axis=0)
            # ground_truth = np.concatenate((ground_truth, ground_truth_labels), axis=0)

    # return predicted, ground_truth
    return F1_list, precision_list, recall_list

def create_vsum(features, mat_file, user):
    y_gt = mat_file['user_score'][:, user]
    num_frames = mat_file['nFrames'].flatten()[0]

    frame_id_list = list(range(num_frames))
    segments = [frame_id_list[i:i + seg_size] for i in range(len(frame_id_list) - seg_size + 1)]
    segments = functools.reduce(lambda x, Y: x + Y, segments)
    x = features[segments]
    enc_x = model(x)
    x = enc_x.data

    features_length = len(x)
    diff = np.abs(len(y_gt) - features_length)

    y_gt = y_gt[diff:]
    num_frames = len(y_gt)
    Y = np.ones(num_frames)

    y_gt = np.squeeze(np.argwhere(y_gt > 0))
    gt = y_gt
    # Approximately equivalent to sampling one frame for every 5 frames selected in the ground truth summary
    # Used to reduce the time for training
    gt = np.squeeze(np.argwhere(y_gt % 5 == 0))
    budget = int((0.15*num_frames)/seg_size)

    # print("Budget size for the summary of video " + str(budget))

    gt_score = np.squeeze(mat_file['gt_score'])[diff:]

    S = VSum(budget, x, Y, gt, gt_score, seg_size)
    return S, y_gt


loss = obj.recall_loss
shells = [obj.representativeness_shell, obj.uniformity_shell, obj.interestingness_shell]
shell_types = [[obj.representativeness_shell], [obj.uniformity_shell], [obj.interestingness_shell],
               [obj.representativeness_shell, obj.uniformity_shell, obj.interestingness_shell]]


max_iterations = 1
for it in range(max_iterations):
    print("Current iteration of learning weights : " + str(it))
    # sample 20 videos to train and find the weights of mixtures
    data_files = [f for f in listdir(directoryPath) if isfile(join(directoryPath, f))]
    np.random.shuffle(data_files)
    for shell_index in range(len(shell_types)):
        if shell_index < 3:
            np.random.shuffle(data_files)
            test_files = data_files
            if shell_index == 0:
                print("-------------------- OBJ : REPRESENTATIVENESS ----------------------------")
                weights = [1, 0, 0]
                F1r_list, pr_list, rr_list = predict(test_files, weights)
                print("STATISTICS: ")
                print("F1 : Mean " + str(np.array(F1r_list).mean()) + ", Variance : " + str(np.array(F1r_list).var()))
                print("Precision : Mean " + str(np.array(pr_list).mean()) + ", Variance : " + str(np.array(pr_list).var()))
                print("Recall : Mean " + str(np.array(rr_list).mean() )+ ", Variance : " + str(np.array(rr_list).var()))
            elif shell_index == 1:
                print("-------------------- OBJ : UNIFORMITY ----------------------------")
                weights = [0, 1, 0]
                F1u_list, pu_list, ru_list = predict(test_files, weights)
                print("STATISTICS: ")
                print("F1 : Mean " + str(np.array(F1u_list).mean()) + ", Variance : " + str(np.array(ru_list).var()))
                print("Precision : Mean " + str(np.array(F1u_list).mean()) + ", Variance : " + str(np.array(ru_list).var()))
                print("Recall : Mean " + str(np.array(F1u_list).mean()) + ", Variance : " + str(np.array(ru_list).var()))
            else:
                print("-------------------- OBJ : INTERESTINGNESS ----------------------------")
                weights = [0, 0, 1]
                F1i_list, pi_list, ri_list = predict(test_files, weights)
                print("STATISTICS: ")
                print("F1 : Mean " + str(np.array(F1i_list).mean()) + ", Variance : " + str(np.array(ri_list).var()))
                print("Precision : Mean " + str(np.array(F1i_list).mean()) + ", Variance : " + str(np.array(ri_list).var()))
                print("Recall : Mean " + str(np.array(F1i_list).mean()) + ", Variance : " + str(np.array(ri_list).var()))
        else:
            print("-------------------- OBJ : COMBINED ----------------------------")
            training_files = data_files[:20]
            test_files = data_files[20:]
            learnt_weights = train(training_files)

            # test on the rest of 5 videos
            F1c_list, pc_list, rc_list = predict(test_files, learnt_weights)
            print("STATISTICS: ")
            print("F1 : Mean " + str(np.array(F1c_list).mean()) + ", Variance : " + str(np.array(rc_list).var()))
            print("Precision : Mean " + str(np.array(F1c_list).mean()) + ", Variance : " + str(np.array(rc_list).var()))
            print("Recall : Mean " + str(np.array(F1c_list).mean()) + ", Variance : " + str(np.array(rc_list).var()))


