import numpy as np
import random
import functools
from chainer import serializers
from os import listdir
from scipy.io import loadmat
from os.path import isfile, join
from Project.videosummarization.gyglivsumexp.script.chain import vid_enc
from Project.videosummarization.gyglivsumexp.script.dataset.vsum import VSum
from Project.videosummarization.gyglivsumexp.submodular_mixtures.utils import utilities
from Project.videosummarization.gyglivsumexp.submodular_mixtures.func import functions
from Project.videosummarization.gyglivsumexp.submodular_mixtures.utils.obj import objectives as obj
from sklearn.metrics import f1_score, precision_score, recall_score


############################################################################################
#   From SUMME dataset we sample few videos for training to find mixture weights
#   We test on other videos and present the summarization results in comparison with user summaries
############################################################################################

seg_size = 5
datasetRoot = '../../data/summe/'
directoryPath = "../../summe/GT/"
model = vid_enc.Model()
serializers.load_npz('data/trained_model/model_par', model)

loss = obj.recall_loss
objective_funcs = [obj.representativeness_shell, obj.uniformity_shell, obj.interestingness_shell]

# Run summarize for 10 times with different rng_seeds inorder to select different set of train videos every time
# Average the F1, precision, recall statistics produced for 10 rounds


def summarize():
    F1_list = []
    precision_list = []
    recall_list = []
    for it in range(10):

        print("Current iteration of learning weights : " + it)

        # sample 20 videos to train and find the weights of mixtures
        data_files = [f for f in listdir(directoryPath) if isfile(join(directoryPath, f))]
        np.random.shuffle(data_files)

        training_files = data_files[:2]
        test_files = data_files[2:]

        learnt_weights = train(training_files)

        # test on the rest of 5 videos
        predicted_summary, ground_summary = predict(test_files, learnt_weights)

        # generate summaries for the rest of the videos and
        # if it == 9:
        #     generate_summary()

        # evaluate the results
        F1, precision, recall = evaluate(predicted_summary, ground_summary)

        F1_list.append(F1)
        precision_list.append(precision)
        recall_list.append(recall)

    return np.sum(F1_list)/10, np.sum(precision)/10, np.sum(recall)/10


def evaluate(predicted, user_summary):
    F1 = f1_score(user_summary, predicted, average="macro")
    precision = precision_score(user_summary, predicted, average="macro")
    recall = recall_score(user_summary, predicted, average="macro")
    return F1, precision, recall


def train(training_files, max_users=3):
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
    learnt_weights, _ = functions.learnSubmodularMixture(training_examples, objective_funcs, loss, params=params)
    print("Finished learning mixture weights")

    return learnt_weights


def predict(test_files, weights):
    predicted = []
    ground_truth = []
    for file_name in test_files:
        video_id = file_name[:-4]
        test_file_mat = loadmat(join(directoryPath, file_name))
        test_features = np.load(datasetRoot + 'feat/vgg/' + video_id + '.npy').astype(np.float32)
        frames, num_users = test_file_mat['user_score'].shape

        random_user_list = random.sample(range(0, num_users), 1)
        for user in random_user_list:
            print("Creating data for " + str(video_id) + ' with user summary ' + str(user))
            S, y_gt = create_vsum(test_features, test_file_mat, user)
            selected, score, minoux_bound = functions.leskovec_maximize(S, weights, objective_funcs, budget=S.budget)
            selected.sort()

            predicted_labels = np.zeros(len(S.Y))
            for idx in list(selected):
                for j in range(seg_size):
                    predicted_labels[idx + j] = 1

            ground_truth_labels = np.zeros(len(S.Y))
            for idx in list(y_gt):
                ground_truth_labels[idx] = 1

            predicted = np.concatenate((predicted, predicted_labels), axis=0)
            ground_truth = np.concatenate((ground_truth, ground_truth_labels), axis=0)

    return predicted, ground_truth


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
    features_length = len(y_gt)
    Y = np.ones(features_length)

    y_gt = np.squeeze(np.argwhere(y_gt > 0))

    # Approximately equivalent to sampling one frame for every 5 frames selected in the ground truth summary
    # Used to reduce the time for training
    gt = np.squeeze(np.argwhere(y_gt % 5 == 0))
    budget = len(gt)
    gt_score = np.squeeze(mat_file['gt_score'])[diff:]
    S = VSum(budget, x, Y, gt, gt_score, seg_size)
    return S, y_gt


print("Started SUMMARIZATION!")
F1, precision, recall = summarize()
print(F1, precision, recall)
print("Finished SUMMARIZATION!")