import numpy as np
from scipy.io import loadmat
from scipy.linalg import block_diag
import itertools
import os
import cv2
import scipy.optimize as opt
import subprocess
import shutil
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


###############################################################
# --------------------- SEQ DPP LINEAR ------------------------
###############################################################


class DataElement:

    def __init__(self):
        pass

    # Some func to be implemented


# Some func and data storage elements
# 1. ids
# 2. grounds
# 3. Ys
# 4. saliency_score
# 5. context_score
# 6. fishers
# 7. YpredSeq
# 8. Ypred
class VideoData(DataElement):

    def __init__(self, mat_data_element, seg_size, dataset):
        self.ids = list(range(len(np.squeeze(mat_data_element[2]))))
        self.nFrames = len(self.ids)
        self.Ys, self.grounds = _gen_seg_(self.nFrames,
                                          _map_ids_(np.squeeze(mat_data_element[2]), np.squeeze(mat_data_element[1])),
                                          seg_size)
        self.videoId = mat_data_element[0][0][0]

        if dataset == 'OVP':
            saliency_score = loadmat(r"./oracle/feature/OVP_v" + str(self.videoId) + '/saliency.mat')['saliency_score']
            context_score = loadmat(r"./oracle/feature/OVP_v" + str(self.videoId) + '/context2.mat')['context_score']
            fishers = loadmat(r"./oracle/feature/OVP_v" + str(self.videoId) + '/fishers_PCA90.mat')['fishers']
        elif dataset == 'Youtube':
            saliency_score = loadmat(r"./oracle/feature/Youtube_v" + str(self.videoId) + '/saliency.mat')[
                'saliency_score']
            context_score = loadmat(r"./oracle/feature/Youtube_v" + str(self.videoId) + '/context2.mat')[
                'context_score']
            fishers = loadmat(r"./oracle/feature/Youtube_v" + str(self.videoId) + '/fishers_PCA90.mat')['fishers']

        self.fts = np.concatenate((saliency_score, context_score, fishers), axis=1)
        self.YpredSeq = []
        self.Ypred = []


def _map_ids_(ids, Y):
    idsY, Y_mapped, Y_ind = np.intersect1d(ids, Y, return_indices=True)
    if len(Y_mapped) is not len(Y):
        print("Error in map_ids: Y is out of ids")
    return Y_mapped


def _gen_seg_(nFrames, gt, seg_size):
    Ys = []
    grounds = []
    nSegments = int(np.ceil(nFrames / seg_size))
    for n in range(nSegments):
        groundsN = list(range(n * seg_size + 1, min((n + 1) * seg_size, nFrames) + 1))
        grounds.append(groundsN)
        YsN = gt[(gt >= groundsN[0]) & (gt <= groundsN[len(groundsN) - 1])]
        Ys.append(YsN)
    return Ys, grounds


def _greedy_sym_(L):
    N = len(L)
    S = []
    # for i in range(N):

    return S


def _initialize_videos_(seg_size, dataset, rng_seed):
    print("Initializing videos")

    # Splitting the training and testing data
    if dataset == 'OVP':
        inds_order = np.random.RandomState(seed=rng_seed).permutation(50)
        inds_tr = inds_order[:40]
        inds_te = inds_order[40:]
        inds_te.sort()
        inds_tr.sort()
    elif dataset == 'Youtube':
        valid_inds = list(range(11, 22)) + list(range(23, 51))
        inds_order = np.random.RandomState(seed=rng_seed).permutation(39)
        inds_tr = inds_order[:31]
        inds_te = inds_order[32:]
        inds_te.sort()
        inds_tr.sort()
        inds_tr = valid_inds[inds_tr]
        inds_te = valid_inds[inds_te]

    videos = []
    if dataset == 'OVP':
        Oracle_OVP = loadmat(r"oracle/Oracle_groundset/Oracle_OVP.mat")
        OVP_linear = loadmat(r"data/OVP_linear.mat")
        Oracle_record = Oracle_OVP['Oracle_record']
        W = OVP_linear['W']
        alpha = np.squeeze(OVP_linear['alpha_'])
    elif dataset == 'Youtube':
        Oracle_YouTube = loadmat(r"oracle/Oracle_groundset/Oracle_Youtube.mat")
        Youtube_linear = loadmat(r"data/Youtube_linear.mat")
        Oracle_record = Oracle_YouTube['Oracle_record']
        W = Youtube_linear['W']
        alpha = np.squeeze(Youtube_linear['alpha_'])

    for t in range(len(Oracle_record)):
        video_data = VideoData(Oracle_record[t], seg_size, dataset)
        if len(video_data.ids) is not len(video_data.fts):
            print("Wrong video " + str(t))
        videos.append(video_data)

    return videos, W, alpha, inds_te, inds_tr


class SeqDppLinear:

    def __init__(self, dataset, rng_seed):
        self.dataset = dataset
        self.rng_seed = rng_seed
        self.seg_size = 10
        self.inf_type = 'exact'
        self.C = np.inf
        self.videos, self.W, self.alpha, self.inds_te, self.inds_tr = _initialize_videos_(self.seg_size, dataset, self.rng_seed)

        if dataset == 'OVP':
            self.OVP_YouTube_index = list(range(21, 71))
        elif dataset == 'Youtube':
            self.OVP_YouTube_index = list(range(11, 21)) + list(range(71, 110))

    # INPUT
    #    videos[k].fts: frames-by-dim
    #    videos[k].grounds: (seg_size)-by-(nFrames/seg_size)
    #    videos[k].Ys: (seg_size)-by-(nFrames/seg_size)
    def train_dpp_linear_MLE(self):
        # Features
        cX = []
        # Ground sets
        cG = []
        # Labeled subsets
        cY = []
        training_videos = [self.videos[i] for i in self.inds_tr]
        for i in range(len(training_videos)):
            cXi = training_videos[i].fts
            cGi = []
            ciGrounds = training_videos[i].grounds
            for j in range(len(ciGrounds)):
                cGij = _map_ids_(training_videos[i].ids, ciGrounds[j])
                cGi.append(cGij)
            cYi = []
            cYs = training_videos[i].Ys
            for k in range(len(cYs)):
                cYik = _map_ids_(training_videos[i].ids, cYs[k])
                cYi.append(cYik)
            cX.append(cXi)
            cY.append(cYi)
            cG.append(cGi)

        _, n = training_videos[0].fts.shape
        m = int(np.size(self.W) / n)

        theta_reg = np.zeros((m, n))

        print("------------------------------------------------------------------------------------------")
        print("OPTIMIZATION BEGINS!!")

        # Minimize hinge loss
        theta0 = np.hstack((self.W.flatten(), self.alpha))
        options = {'disp':99, 'maxiter':500, 'maxfun': 200000, 'gtol': 1e-10, 'ftol': 1e-10, 'iprint':99}
        result = opt.minimize(self._compute_fg_, theta0, tol=1e-6, args=(cX, cG, cY, self.C, theta_reg), method='L-BFGS-B', options=options)

        print("OPTIMIZATION ENDS!!")
        print("------------------------------------------------------------------------------------------")

        theta = result.x

        np.savetxt('theta.csv', np.array(theta), delimiter=',')

        # Recover W, V, alpha
        alpha = theta[len(theta)-1]
        print("Found alpha :" + str(alpha))
        alpha = max(theta[len(theta)-1], 1e-6)
        W = theta[:len(theta)-1]
        W = np.reshape(W, (m, n))

        fval = self._compute_fg_(theta, cX, cG, cY, self.C, theta_reg)

        return W, alpha, fval

    def _compute_fg_(self, theta, cX, cG, cY, C, theta_reg):

        alpha = max(theta[len(theta) - 1], 1e-6)
        theta = theta[:len(theta) - 1]

        n = cX[1].shape[1]
        m = int(len(theta) / n)
        W = np.reshape(theta, (m, n))

        # print("Processing videos : ")

        f = 0
        # GW = np.zeros((m, n))
        galpha = 0
        for k in range(len(cX)):
            [fk, gWk, gAk] = self._compute_fg_one_data_(W, alpha, cX[k], cG[k], cY[k])
            f = f - fk
            # GW = GW - gWk
            # galpha = galpha - gAk

        # g = GW
        # g = np.hstack((np.array(g).flatten(), galpha[0, 0]))

        if C != np.inf:
            diff = W.flatten() - theta_reg.flatten()
            f = C * f + 0.5 * (diff.conj().transpose() @ diff)
            # g = C * g + np.hstack((diff, 0))

        return f

    #  INPUT:
    #  theta: the parameter to learn
    #  X: #frames-by-dim, features
    #  Gs: ground sets, cell, 1-by-T
    #  Ys: labeled subsets, cell, 1-by-T
    def _compute_fg_one_data_(self, W, alpha, X, Gs, Ys):
        Gs = [[]] + Gs
        Ys = [[]] + Ys
        [m, n] = W.shape
        WX = np.dot(W, X.conj().transpose())
        LL = np.dot(WX.conj().transpose(), WX)

        f = 0
        gW = np.zeros((m, n))
        galpha = 0
        for t in range(1, len(Gs)):
            if len(Ys[t-1]) == 0:
                Y = np.array(Ys[t]).flatten()
                V = np.array(Gs[t]).flatten()
            else:
                Y = np.concatenate((np.array(Ys[t - 1]).flatten(), np.array(Ys[t]).flatten()), axis=0)
                V = np.concatenate((np.array(Ys[t - 1]).flatten(), np.array(Gs[t]).flatten()), axis=0)
            VY, V_ind, Y_ind = np.intersect1d(V, Y, return_indices=True)
            Y = V_ind
            L = LL[np.ix_(V, V)] + alpha * np.eye(len(V))

            # Fix numerical issues
            L = (L + L.conj().transpose()) / 2
            Ysz = np.zeros((len(Ys[t - 1]), len(Ys[t - 1])))
            Gso = np.eye(len(Gs[t]))
            Iv = block_diag(Ysz, Gso)

            # Compute function value
            J = np.log(np.linalg.det(L[np.ix_(Y, Y)])) - np.log(np.linalg.det(L + Iv))

            # Compute gradients
            # [g, gA] = self._compute_g_(W, X[0:len(V), :], L, Y, Iv)

            # overall
            f = f + J
            # gW = gW + g
            # galpha = galpha + gA

        return f, gW, galpha

    # gradient
    def _compute_g_(self, W, X, L, Y, Iv):
        # partial J / partial
        Ainv = np.linalg.inv(L + Iv)

        if np.size(Y) == 0:
            LYinv = 0
        else:
            LYinv = np.zeros((len(L), len(L)))
            LYinv[0:len(Y), 0:len(Y)] = np.linalg.inv(L[0:len(Y), 0:len(Y)])

        gL = LYinv - Ainv
        g = 2 * np.dot(np.dot(np.dot(W, X.conj().transpose()), gL), X)
        gA = np.matrix(gL).trace()

        return g, gA

    def test_dpp_inside(self, videos, W, alpha, inf_type):
        Ls = []
        for i in range(len(videos)):
            print("TestDPP : " + str(i))
            [videos[i].YpredSeq, videos[i].Ypred, Lsi] = self._predictY_inside_(videos[i], W, [], alpha, inf_type)
            Ls.append(Lsi)
        return videos, Ls

    def _predictY_inside_(self, video, W, L, alpha, inf_type):

        if len(L) == 0:
            X = video.fts
            WW = np.dot(W.conj().transpose(), W)
            L = np.dot(X, np.dot(WW, X.conj().transpose())) + alpha * np.identity(len(X))

        # Correct for numerical errors
        L = (L + L.conj().transpose()) / 2
        # For dummy video segment
        Gs = [[]]
        for n in range(len(video.grounds)):
            Gsn = _map_ids_(video.ids, video.grounds[n])
            Gs.append(Gsn)

        # For recording predicted results
        Y = [[]]
        Y_record = []
        for t in range(1, len(Gs)):
            if len(Y[t-1]) == 0:
                V = Gs[t]
            else:
                V = np.concatenate((Y[t - 1], Gs[t]), axis=0)
            VY, V_ind, Y_ind_t = np.intersect1d(V, Y[t - 1], return_indices=True)
            Y_loc = V_ind
            VGs, V_ind, Gs_ind_t = np.intersect1d(V, Gs[t], return_indices=True)
            Gs_loc = V_ind

            L_window = L[np.ix_(V, V)]
            Yz = np.zeros((len(Y[t - 1]), len(Y[t - 1])))
            Gso = np.eye(len(Gs[t]))
            Iv = block_diag(Yz, Gso)

            if inf_type == 'exact':
                whichs = []
                a_list = list(range(0, len(Gs[t])))
                for r in range(1, min(11, len(Gs[t])) + 1):
                    combinations_object = itertools.combinations(a_list, r)
                    combinations_list = list(combinations_object)
                    whichs += combinations_list

                # Starting with empty set
                best_comb = -1
                best_J = np.linalg.det(L_window[np.ix_(Y_loc, Y_loc)]) / np.linalg.det(L_window + Iv)

                # Brute force for each subset of Gs[t]
                for at_comb in range(len(whichs)):
                    YG_loc = np.concatenate((Y_loc, [Gs_loc[i] for i in list(whichs[at_comb])]), axis=0)
                    J = np.linalg.det(L_window[np.ix_(YG_loc, YG_loc)]) / np.linalg.det(L_window + Iv)
                    if J > best_J:
                        best_J = J
                        best_comb = at_comb

                Y_t = []
                if best_comb != -1:
                    Y_t = np.array([V[j] for j in [Gs_loc[i] for i in list(whichs[best_comb])]])
                    Y.append(Y_t)
                    if len(Y_record) == 0:
                        Y_record = Y_t.conj().transpose()
                    else:
                        Y_record = np.concatenate((Y_record, Y_t.conj().transpose()), axis=0)
                else:
                    Y.append(Y_t)
            # else:
            #     L_cond = np.linalg.inv(L_window + Iv)
            #     L_cond = np.linalg.inv(L_cond[0:len(Gs_loc)][0:len(Gs_loc)]) - np.eye(len(Gs_loc))
            #     select_loc_Gs = _greedy_sym_(L_cond)
            #     Y[t] = V[Gs_loc[select_loc_Gs]]
            #     if not bool(Y_record):
            #         Y_record = V[Gs_loc[select_loc_Gs]].T
            #     else:
            #         Y_record = np.concatenate((Y_record, V[Gs_loc[select_loc_Gs]].T), axis=1)

        Y_record = np.sort(np.unique(Y_record))
        return Y, Y_record, L

    def seqDPP_evaluate(self, videos_te, inds_te, num_order, approach_name):
        self._kill_seqDPP_(approach_name, inds_te)
        F1, precision, recall = self._make_folder_oracle_(videos_te, approach_name, inds_te)
        return F1, precision, recall
        # approaches_list = [approach_name]
        # O_S_detail, _ = self._comparison_seqDPP_(videos_te, approaches_list, num_order, inds_te)
        # _, True_RP, True_F1 = self._sample_seqdpp_comp_(O_S_detail)
        # res = np.hstack((True_F1, True_RP))
        # return res

    def _kill_seqDPP_(self, approach_name, inds_te):
        folderName = './data/OVP_YouTube_cmp'
        for n in range(len(inds_te)):
            videoName = 'v' + str(self.OVP_YouTube_index[inds_te[n]])
            if os.path.isdir(folderName + '/' + videoName + '/' + approach_name):
                print("Removed : " + folderName + '/' + videoName + '/' + approach_name)
                shutil.rmtree(folderName + '/' + videoName + '/' + approach_name)

    def _make_folder_oracle_(self, videos, approach_name, inds_te):

        folderName = './data/OVP_YouTube_cmp'
        framePath = './data/Frames_sampled/'

        if self.dataset == 'OVP':
            Oracle_OVP = loadmat(r"oracle/Oracle_groundset/Oracle_OVP.mat")
            Oracle_record = Oracle_OVP['Oracle_record']
            framePath += 'OVP/'
        elif self.dataset == 'YouTube':
            Oracle_YouTube = loadmat(r"oracle/Oracle_groundset/Oracle_Youtube.mat")
            Oracle_record = Oracle_YouTube['Oracle_record']
            framePath += 'Youtube/'

        predicted = []
        user_summary = []
        user_index = list(range(1, 6))
        for n in range(len(inds_te)):
            videoName = 'v' + str(self.OVP_YouTube_index[inds_te[n]])
            vidFrame = loadmat(os.path.join(framePath, videoName + '.mat'))['vidFrame']
            inds_frame = Oracle_record[inds_te[n]][2]
            nrFramesTotal = vidFrame[0][0][2][0][0]
            frames_inds_frame_int, frames_int, inds_frame_int = np.intersect1d(15 * np.array(list(range(1, nrFramesTotal + 1))), np.sort(inds_frame),
                                        return_indices=True)

            gt_frame = Oracle_record[inds_te[n]][1]
            frames_gt_frame_int, frames_inter, gt_frame_int = np.intersect1d(
                15 * np.array(list(range(1, nrFramesTotal + 1))), np.sort(inds_frame),
                return_indices=True)

            directPath = folderName + '/' + videoName + '/' + approach_name
            os.makedirs(directPath)

            frame_index = [frames_inds_frame_int[i] for i in videos[n].Ypred]
            # frame_index = inds_frame[videos[n].Ypred]
            for m in range(len(frame_index)):
                cv2.imwrite(directPath + '/Frame' + str(frame_index[m]) + '.jpeg',
                            vidFrame[0][0][3][0][int(frame_index[m]/15)-1][0])

            model = 'both'
            if model == 'u':
                for u in user_index:
                    userName = 'user' + str(u)
                    userSummaryDir = folderName + '/' + videoName + '/' + userName
                    userSummaryFrameStrings = [f for f in os.listdir(userSummaryDir)]
                    userSummaryFrames = [int(s.replace("Frame", "").replace(".jpeg", "")) for s in userSummaryFrameStrings]

                    videoPredLabels, userSummaryLabels = self._generate_labels_(predicted_ind=frame_index, user_summary_ind=userSummaryFrames,
                                                                                vid_frames_count=nrFramesTotal)
                    predicted = np.concatenate((predicted, videoPredLabels), axis=0)
                    user_summary = np.concatenate((user_summary, userSummaryLabels), axis=0)
            elif model == 'gt':
                videoPredLabels, userSummaryLabels = self._generate_labels_(predicted_ind=frame_index,
                                                                            user_summary_ind=frames_gt_frame_int,
                                                                            vid_frames_count=nrFramesTotal)
                predicted = np.concatenate((predicted, videoPredLabels), axis=0)
                user_summary = np.concatenate((user_summary, userSummaryLabels), axis=0)
            elif model == 'both':
                for u in user_index:
                    if np.random.random() > 0.5:
                        continue

                    userName = 'user' + str(u)
                    userSummaryDir = folderName + '/' + videoName + '/' + userName
                    userSummaryFrameStrings = [f for f in os.listdir(userSummaryDir)]
                    userSummaryFrames = [int(s.replace("Frame", "").replace(".jpeg", "")) for s in userSummaryFrameStrings]

                    videoPredLabels, userSummaryLabels = self._generate_labels_(predicted_ind=frame_index, user_summary_ind=userSummaryFrames,
                                                                                vid_frames_count=nrFramesTotal)
                    predicted = np.concatenate((predicted, videoPredLabels), axis=0)
                    user_summary = np.concatenate((user_summary, userSummaryLabels), axis=0)

                videoPredLabels, userSummaryLabels = self._generate_labels_(predicted_ind=frame_index,
                                                                            user_summary_ind=frames_gt_frame_int,
                                                                            vid_frames_count=nrFramesTotal)
                predicted = np.concatenate((predicted, videoPredLabels), axis=0)
                user_summary = np.concatenate((user_summary, userSummaryLabels), axis=0)


        F1, precision, recall = self._compute_stats_(predicted, user_summary)

        return F1, precision, recall

    def _generate_labels_(self, predicted_ind, user_summary_ind, vid_frames_count):

        predicted_labels = np.zeros(15 * (vid_frames_count + 1))
        user_summary_labels = np.zeros(15 * (vid_frames_count + 1))

        # For every selected frame, we consider the next 5 frames as selected ones to produces skims
        for pri in predicted_ind:
            for j in range(5):
                predicted_labels[pri+j] = 1

        for usi in user_summary_ind:
            for j in range(5):
                user_summary_labels[usi+j] = 1

        return predicted_labels, user_summary_labels


    def _compute_stats_(self, predicted, user_summary):
        F1 = f1_score(user_summary, predicted, average="macro")
        precision = precision_score(user_summary, predicted, average="macro")
        recall = recall_score(user_summary, predicted, average="macro")
        return F1, precision, recall

    # def _comparison_seqDPP_(self, approaches_list, order_cmp, inds_te):
    #
    #     # Setup
    #     self._write_seqDPP_input_('input_wholeCmp' + str(order_cmp) + '.txt', approaches_list, inds_te)
    #
    #     threshold = 0.5
    #     user_index = list(range(1, 6))
    #     OVP_YouTube_index = [self.OVP_YouTube_index[i] for i in inds_te]
    #
    #     # VSUMM Eval
    #     order_cmp_str = str(order_cmp)
    #
    #     # Constructing Labels for predicted vs user summaries to evaluate F1, Recall, Precision
    #     input_file = 'input_wholeCmp' + str(order_cmp) + '.txt'
    #
    #     f_inp = open(input_file, 'r')
    #
    #
    #
    #
    #
    #     run_command = '-jar CUS.jar -i input_wholeCmp' + order_cmp_str + '.txt -o output_wholeCmp' + order_cmp_str + '.txt -u ' + str(len(user_index)) + ' -a ' + str(len(approaches_list)) + ' -t ' + str(threshold)
    #
    #     subprocess.call('java ' + run_command, shell=True)
    #     subprocess.check_output(['icacls.exe', 'input_wholeCmp' + order_cmp_str + '.txt', '/GRANT', 'everyone:(f)'],
    #                             stderr=subprocess.STDOUT)
    #     subprocess.check_output(['icacls.exe', 'output_wholeCmp' + order_cmp_str + '.txt', '/GRANT', 'everyone:(f)'],
    #                             stderr=subprocess.STDOUT)
    #     os.system(
    #         '/usr/lib/jvm/jre-1.6.0/bin/java -jar CUS' + order_cmp_str + '.jar -i input_wholeCmp' + order_cmp_str + '.txt -o output_wholeCmp' + order_cmp_str +
    #         '.txt -u ' + str(len(user_index)) + ' -a ' + str(len(approaches_list)) + ' -t ' + str(threshold))
    #     os.system('chmod 777 input_wholeCmp' + order_cmp_str + '.txt -R')
    #     os.system('chmod 777 output_wholeCmp' + order_cmp_str + '.txt -R')
    #
    #     output_record, output_summary = self._read_seqDPP_output_('output_wholeCmp' + str(order_cmp) + '.txt',
    #                                                               OVP_YouTube_index, len(user_index),
    #                                                               len(approaches_list))
    #     os.remove('input_wholeCmp' + order_cmp_str + '.txt')
    #     os.remove('output_wholeCmp' + order_cmp_str + '.txt')
    #     return output_record, output_summary

    def _write_seqDPP_input_(self, filename, approaches_list, inds_te):
        user_index = list(range(1, 6))
        folderName = './data/OVP_YouTube_cmp'

        fid = open(filename, 'w')
        fid.truncate(0)

        if os.stat(filename).st_size == 0:
            print("Truncated file")

        for k in range(len(inds_te)):
            videoName = 'v' + str(self.OVP_YouTube_index[inds_te[k]])
            fid.write(folderName + '/' + videoName + '\n')
            for n in range(len(user_index)):
                userName = 'user' + str(user_index[n])
                fid.write(folderName + '/' + videoName + '/' + userName + '\n')
            for n in range(len(approaches_list)):
                fid.write(folderName + '/' + videoName + '/' + approaches_list[n] + '\n')
            fid.write('\n')

        fid.close()
        # subprocess.check_output(['icacls.exe', filename, '/GRANT', 'everyone:(f)'], stderr=subprocess.STDOUT)
        return

    def _read_seqDPP_output_(self, file_name, video_index, num_user, num_approach):

        output_record = []
        output_summery = np.zeros((num_approach, 2))

        fid = open(file_name)
        # now_line = fid.

        # TODO: Fix this

        pass

    def _sample_seqdpp_comp_(self, O_S_detail):
        NUM_2 = len(O_S_detail)
        True_CU = np.zeros(NUM_2, 2)
        True_RP = np.zeros(NUM_2, 2)
        True_F1 = np.zeros(NUM_2, 1)
        for num2 in range(NUM_2):
            now_OS2 = O_S_detail[num2][:]
            for now_at in range(len(now_OS2)):
                now_user_CU = now_OS2[now_at]
                per_video_CU = np.sum(now_user_CU, axis=0) / len(now_user_CU)
                now_user_CU_col_sum = np.sum(now_user_CU, axis=1) + 1e-10
                per_video_true_p = np.sum(now_user_CU.flatten() / now_user_CU_col_sum, axis=0) / len(now_user_CU)
                per_video_true_F1 = np.sum((now_user_CU.flatten() * now_user_CU.flatten() / now_user_CU_col_sum) /
                                           now_user_CU.flatten() + (now_user_CU.flatten() / now_user_CU_col_sum),
                                           axis=0) / len(now_user_CU)
                True_CU[num2][:] = True_CU[num2][:] + per_video_CU / len(now_OS2)
                # TODO: Fix this
                # True_RP[num2][:] = True_RP[num2][:] +
                True_F1[num2] = True_F1[num2] + per_video_true_F1 / len(now_OS2)
        return True_CU, True_RP, True_F1

    def train(self):
        print("Started Training!")
        pass

    def test(self):
        print("Started generating summaries!")
        pass

    def evaluate(self):
        print("Started evaluation!")
        pass
