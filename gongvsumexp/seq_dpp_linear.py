import numpy as np
from scipy.io import loadmat
from scipy.linalg import block_diag
import itertools
import os
import cv2

###############################################################
# --------------------- SEQ DPP LINEAR ------------------------
###############################################################


class DataElement:

    def __init__(self):
        pass

    # Some functions to be implemented


    # Some functions and data storage elements
    # 1. ids
    # 2. grounds
    # 3. Ys
    # 4. saliency_score
    # 5. context_score
    # 6. fishers
    # 7. YpredSeq
    # 8. Ypred
class VideoData(DataElement):

    def __init__(self, mat_data_element, seg_size):
        self.ids = mat_data_element[2]
        self.nFrames = len(self.ids)
        self.Ys, self.grounds = _gen_seg_(self.nFrames, _map_ids_(self.ids, mat_data_element[1]), seg_size)
        self.videoId = mat_data_element[0][0][0]
        saliency_score = loadmat(r"./oracle/feature/OVP_v" + str(self.videoId) + '/saliency.mat')
        context_score = loadmat(r"./oracle/feature/OVP_v" + str(self.videoId) + '/context2.mat')
        fishers = loadmat(r"./oracle/feature/OVP_v" + str(self.videoId) + '/fishers_PCA90.mat')
        self.fts = np.concatenate((saliency_score, context_score, fishers), axis=1)
        self.YpredSeq = []
        self.Ypred = []





def _map_ids_(ids, Y):
    idsY, Y_mapped, Y_ind = np.intersect1d(ids, Y, return_indices=True)
    if len(Y_mapped) is not len(Y):
        print("Error in map_ids: Y is out of ids")
    return Y_mapped


def _gen_seg_(nFrm, gt, seg_size):
    nSeg = np.ceil(nFrm/seg_size)
    Ys = []
    grounds = []
    for n in range(nSeg):
        groundsn = list(range(n*seg_size+1, min(n*seg_size, nFrm)))
        grounds.append(groundsn)
        Ysn = gt[ (gt >= groundsn[0]) & (gt <= groundsn[len(groundsn)-1]) ]
        Ys.append(Ysn)
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
    elif dataset == 'YouTube':
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
        alpha_ = OVP_linear['alpha_']
        for t in range(len(Oracle_record)):
            new_videos_dict = {}
            new_videos_dict['ids'] = Oracle_record[t][2]
            new_videos_dict['Ys'], videos['grounds'] = _gen_seg_(len(Oracle_record[t][2][0]), _map_ids_(Oracle_record[t][3][0], Oracle_record[t][2][0]), seg_size)
            saliency_score = loadmat(r"./oracle/feature/OVP_v" + str(Oracle_record[t][0][0][0])+'/saliency.mat')
            context_score = loadmat(r"./oracle/feature/OVP_v" + str(Oracle_record[t][0][0][0]) + '/context2.mat')
            fishers = loadmat(r"./oracle/feature/OVP_v" + str(Oracle_record[t][0][0][0]) + '/fishers_PCA90.mat')
            new_videos_dict['fts'] = np.concatenate((saliency_score, context_score, fishers), axis = 1)
            videos.append(new_videos_dict)


            # TODO: Add length check as done in the code
            # if len(Oracle_record[t][3][0]) not len(sa)

    elif dataset == 'YouTube':
        Oracle_YouTube = loadmat(r"oracle/Oracle_groundset/Oracle_Youtube.mat")
        YouTube_linear = loadmat(r"data/YouTube_linear.mat")
        Oracle_record = Oracle_YouTube['Oracle_record']
        W = YouTube_linear['W']
        alpha_ = YouTube_linear['alpha_']
        for t in range(len(Oracle_record)):
            new_videos_dict = {}
            new_videos_dict['ids'] = Oracle_record[t][3]
            new_videos_dict['Ys'], videos['grounds'] = _gen_seg_(len(Oracle_record[t][3][0]),
                                                                          _map_ids_(Oracle_record[t][3][0],
                                                                                         Oracle_record[t][2][0]),
                                                                          seg_size)
            saliency_score = loadmat(r"./oracle/feature/Youtube_v" + str(Oracle_record[t][0][0][0]) + '/saliency.mat')
            context_score = loadmat(r"./oracle/feature/Youtube_v" + str(Oracle_record[t][0][0][0]) + '/context2.mat')
            fishers = loadmat(r"./oracle/feature/Youtube_v" + str(Oracle_record[t][0][0][0]) + '/fishers_PCA90.mat')
            new_videos_dict['fts'] = np.concatenate((saliency_score, context_score, fishers), axis = 1)
            videos.append(new_videos_dict)
            # TODO: Add length check as done in the code
            # if len(Oracle_record[t][3][0]) not len(sa)

    for t in range(len(videos)):
        videos[t]['YpredSeq'] = []
        videos[t]['Ypred'] = []

    return videos, W, alpha_


class seqDppLinear:

    def __init__(self, dataset, rng_seed):
        self.dataset = dataset
        self.rng_seed = rng_seed
        self.seg_size = 10
        self.inf_type = 'exact'
        self.C = np.inf



    def _train_dpp_linear_MLE_(self, videos, C, W0, alpha_0):

        # INPUT
        #    videos(k).fts: matrix, #frames-by-dim
        #    videos(k).grounds: cell, 1-by-T
        #    videos(k).Ys: cell, 1-by-T

        # Features
        cX = []
        # Ground sets
        cG = []
        # Labeled subsets
        cY = []

        for i in range(len(videos)):
            cXi = videos[i].fts
            cGi = []
            cigrounds = videos[i].grounds
            for j in range(len(cigrounds)):
                cGij = _map_ids_(videos[i].ids, cigrounds[j])
                cGi.append(cGij)
            cYi = []
            cYs = videos[i].Ys
            for k in range(len(cYs)):
                cYik = _map_ids_(videos[i].ids, cYs[k])
                cYi.append(cYik)
            cX.append(cXi)
            cY.append(cYi)
            cG.append(cGi)

        _, n = videos[0].fts
        m = np.size(W0)/n

        theta_reg = np.zeros((m,n))

        # Minimize hinge loss


        # TODO: Fix this
        # # Recover W, V, alpha
        # W = theta[:len(theta)-1]
        # W = np.reshape(W, m, n)
        # alpha = max(theta[-1], 1e-6)
        #
        # return W, alpha, fval, exitflag, output

    def _compute_fg_(self, theta, cX, cG, cY, C, theta_reg):

        alpha = max(theta[-1], 1e-6)
        theta = theta[:len(theta)-1]

        n = cX[1].shape[1]
        m = len(theta)/n
        W = np.reshape(theta, m, n)

        f = 0
        GW = np.zeros((m,n))
        galpha = 0
        for k in range(len(cX)):
            print("Process video : " + str(k))
            [fk, gWk, gAk] = self._compute_fg_one_data_(W, alpha, cX[k], cG[k], cY[k])
            f = f-fk
            GW = GW - gWk
            galpha = galpha - gAk

        g = GW
        g = np.hstack((g.flatten(), galpha))

        if C is not np.inf:
            diff = W.flatten() - theta_reg.flatten()
            f = C @ f + 0.5 * (diff.conj().transpose() @ diff)
            g = C @ g + np.vstack((diff, 0))

        return f, g

    def _compute_fg_one_data_(self, W, alpha, X, Gs, Ys):
        #  input:
        #  theta: the parameter to learn
        #  X: #frames-by-dim, features
        #  Gs: ground sets, cell, 1-by-T
        #  Ys: labeled subsets, cell, 1-by-T

        Gs = [[]] + Gs
        Ys = [[]] + Ys
        [m, n] = W.shape
        WX = np.dot(W, X.conj().transpose())
        LL = np.dot(WX.conj().transpose(), WX)

        f = 0
        gW = np.zeros((m,n))
        galpha = 0
        for t in range(1,len(Gs)):
            Y = np.vstack((Ys[t-1].flatten(), Ys[t].flatten()))
            V = np.vstack((Ys[t-1].flatten(), Gs[t].flatten()))
            VY, V_ind, Y = np.intersect1d(V, Y, return_indices=True)
            L = LL[0:len(V)][0:len(V)] + alpha * np.eye(len(V))

            # Fix numerical issues
            L = ( L + L.conj().transpose() )/2
            Ysz = np.zeros(len(Ys[t - 1]))
            Gso = np.eye(len(Gs[t]))
            Iv = block_diag(Ysz, Gso)

            # Compute function value
            J = np.log(np.linalg.det(L[0:len(Y)][0:len(Y)])) - np.log(np.linalg.det(L + Iv))

            # Compute gradients
            [g, gA] = self._compute_g_(W, X[0:len(V)][:], L, Y, Iv)

            # overall
            f = f + J
            gW = gW + g
            galpha = galpha + gA

        return f, gW, galpha

    def _compute_g_(self, W, X, L, Y, Iv):
        # gradient
        # partial J / partial L
        Ainv = np.linalg.inv(L + Iv)

        if not bool(Y):
            LYinv = 0
        else:
            LYinv = np.zeros(len(L))
            LYinv[0:len(Y)][0:len(Y)] = np.linalg.inv(L[0:len(Y)][0:len(Y)])

        gL = LYinv - Ainv
        g = 2 * np.dot(np.dot(np.dot(W, X.conj().transpose()), gL), X)
        gA = np.matrix(gL).trace()

        return g, gA


    def _test_dpp_inside_(self, videos, W, alpha, Ls, inf_type):
        Ls = []
        for i in range(len(videos)):
            print("TestDPP : " + str(i))
            [videos[i].YpredSeq, videos[i].Ypred, Lsi] = self._predictY_inside_(videos(i), W, [], alpha, inf_type)
            Ls.append(Lsi)
        return videos, Ls

    def _predictY_inside_(self, video, W, L, alpha, inf_type):

        # TODO : For all matrix transpose do conjugate transpose
        if not bool(L):
            X = video.fts
            WW = np.dot(W.T, W)
            L = np.dot(X, np.dot(WW, X.T)) + alpha * np.identity(len(X))

        # Correct for numerical errors
        L = (L + L.T)/2
        # For dummy video segment
        Gs = [[]]
        for n in range(len(video.grounds)):
            Gsn = _map_ids_(video.ids, video.grounds[n])
            Gs.append(Gsn)

        # For recording predicted results
        Y = []
        Y_record = []
        for t in range(1, len(Gs)):
            V = np.concatenate((Y[t-1], Gs[t]), axis=0)
            VY, V_ind, Y_loc = np.intersect1d(V, Y[t-1], return_indices=True)
            VGs, V_ind, Gs_loc = np.intersect1d(V, Gs[t-1], return_indices=True)

            L_window = L[0:len(V)][0:len(V)]
            Yz = np.zeros(len(Y[t-1]))
            Gso = np.eye(len(Gs[t]))
            Iv = block_diag(Yz, Gso)

            if inf_type == 'exact':
                whichs = []
                a_list = list(range(1, len(Gs[t])))
                for r in range(min(10, len(Gs[t]))+1):
                    combinations_object = itertools.combinations(a_list, r)
                    combinations_list = list(combinations_object)
                    whichs += combinations_list

                # Starting with empty set
                best_comb = 0
                best_J = np.linalg.det(L_window[0:len(Y_loc)][0:len(Y_loc)]) / np.linalg.det(L_window + Iv)

                # Brute force for each subset of Gs[t]
                for at_comb in range(len(whichs)):
                    YG_loc = np.concatenate((Y_loc, Gs_loc[whichs[at_comb]]))
                    J = np.linalg.det(L_window[0:len(YG_loc)][0:len(YG_loc)])/ np.linalg.det(L_window+Iv)
                    if J > best_J:
                        best_J = J
                        best_comb = at_comb

                if best_comb is not 0:
                    Y[t] = V[Gs_loc[whichs[best_comb]]]
                    if not bool(Y_record):
                        Y_record = V[Gs_loc[whichs[best_comb]]].T
                    else:
                        Y_record = np.concatenate((Y_record, V[Gs_loc[select_loc_Gs]].T), axis=1)
            else:
                L_cond = np.linalg.inv(L_window + Iv)
                L_cond = np.linalg.inv(L_cond[0:len(Gs_loc)][0:len(Gs_loc)]) - np.eye(len(Gs_loc))
                select_loc_Gs = _greedy_sym_(L_cond)
                Y[t] = V[Gs_loc[select_loc_Gs]]
                if not bool(Y_record):
                    Y_record = V[Gs_loc[select_loc_Gs]].T
                else:
                    Y_record = np.concatenate((Y_record, V[Gs_loc[select_loc_Gs]].T), axis=1)

        Y_record = np.sort(np.unique(Y_record))
        return Y, Y_record, L

    def _seqDPP_evaluate(self, videos_te, inds_te, num_order, approach_name, dataset):
        self._kill_seqDPP_(approach_name, inds_te, dataset)
        self._make_folder_oracle_(videos_te, approach_name, inds_te, dataset)
        O_S_detail, _ = self._comparison_seqDPP_(approach_name, num_order,inds_te, dataset)
        _, True_RP, True_F1 = self._sample_seqdpp_comp_(O_S_detail)
        res = np.hstack((True_F1, True_RP))
        return res

    def _kill_seqDPP_(self, appraoch_name, te_inds, dataset):
        foldername = '../data/OVP_YouTube_cmp'
        if dataset == 'OVP':
            OVP_YouTube_index = list(range(21,71))
        elif dataset == 'YouTube':
            OVP_YouTube_index = list(range(11,21)) + list(range(71, 110))
        for n in range(len(te_inds)):
            videoname = 'v' + OVP_YouTube_index[te_inds[n]]
            if os.path.isdir(foldername + '/' + videoname + '/' + appraoch_name):
                print("Removed : " + foldername + '/' + videoname + '/' + appraoch_name)
                os.remove(foldername + '/' + videoname + '/' + appraoch_name)

    def _make_folder_oracle_(self, videos, approach_name, te_inds, dataset):
        foldername = '../data/OVP_YouTube_cmp'
        framePath = '../data/Frames_sampled'

        if dataset == 'OVP':
            Oracle_OVP = loadmat(r"oracle/Oracle_groundset/Oracle_OVP.mat")
            Oracle_record = Oracle_OVP['Oracle_record']
            OVP_YouTube_index = list(range(21, 71))
        elif dataset == 'YouTube':
            Oracle_YouTube = loadmat(r"oracle/Oracle_groundset/Oracle_Youtube.mat")
            Oracle_record = Oracle_YouTube['Oracle_record']
            OVP_YouTube_index = list(range(11, 21)) + list(range(71, 110))

        for n in range(len(te_inds)):
            videoname = 'v' + OVP_YouTube_index[te_inds[n]]
            vidFrame = loadmat(os.path.join(framePath, videoname + '.mat'))
            inds_frame = Oracle_record[te_inds[n]][2]
            inds_frame = np.intersect1d( 15*list(range(1, vidFrame.nrFramesTotal)), np.sort(inds_frame), return_indices=True)

            directPath = foldername + '/' + videoname + '/' + approach_name
            os.mkdir(directPath)

            frame_index = inds_frame[videos[n].Ypred]
            for m in range(len(frame_index)):
                cv2.imwrite(directPath + '/Frame' + str(frame_index[m]) + '.jpeg', vidFrame['frames'][frame_index[m]/15]['cdata'])

            os.system('chmod 777 ' + directPath + '-R')

        return

    def _comparison_seqDPP_(self, approach_name, order_cmp, te_inds, dataset):
        # Setup
        self._write_seqDPP_input_('input_wholeCmp' + str(order_cmp) + '.txt', approach_name, te_inds, dataset)
        threshold = 0.5
        user_index = list(range(1,6))
        if dataset == 'OVP':
            OVP_YouTube_index = list(range(21, 71))
        elif dataset == 'YouTube':
            OVP_YouTube_index = list(range(11, 21)) + list(range(71, 110))
        OVP_YouTube_index = OVP_YouTube_index[te_inds]
        # VSUMM Eval
        order_cmp_str = str(order_cmp)
        os.system('/usr/lib/jvm/jre-1.6.0/bin/java -jar CUS'+ order_cmp_str +'.jar -i input_wholeCmp' + order_cmp_str + '.txt -o output_wholeCmp' + order_cmp_str +
    '.txt -u ' + str(len(user_index)) + ' -a ' + str(len(approach_name)) + ' -t ' + str(threshold) )
        os.system('chmod 777 input_wholeCmp' + order_cmp_str + '.txt -R')
        os.system('chmod 777 output_wholeCmp' + order_cmp_str + '.txt -R')
        output_record, output_summary = self._read_seqDPP_output_('output_wholeCmp' + str(order_cmp) + '.txt', OVP_YouTube_index, len(user_index), len(approach_name))
        os.remove('input_wholeCmp' + order_cmp_str + '.txt')
        os.remove('output_wholeCmp' + order_cmp_str + '.txt')
        return output_record, output_summary

    def _write_seqDPP_input_(self, filename, approach_name, te_inds, dataset):
        if dataset == 'OVP':
            OVP_YouTube_index = list(range(21, 71))
        elif dataset == 'YouTube':
            OVP_YouTube_index = list(range(11, 21)) + list(range(71, 110))

        user_index = list(range(1, 6))
        foldername = '../data/OVP_YouTube_cmp'

        fid = open(filename, 'w')

        for k in range(len(te_inds)):
            videoname = 'v' + OVP_YouTube_index[te_inds[k]]
            fid.write(videoname + '\n' + foldername + '/')
            for n in range(len(user_index)):
                userName = 'user' + user_index[n]
                fid.write(videoname + ' ' + userName + '\n' + foldername + '/')
            for n in range(len(approach_name)):
                fid.write(videoname + ' ' + approach_name[n] + '\n' + foldername + '/')
            fid.write('\n')

        fid.close()
        os.system('chmod 777 ' + filename + ' -R')
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
                per_video_CU = np.sum(now_user_CU, axis=0)/len(now_user_CU)
                now_user_CU_col_sum = np.sum(now_user_CU, axis=1) + 1e-10
                per_video_true_p = np.sum( now_user_CU.flatten()/now_user_CU_col_sum ,axis = 0)/len(now_user_CU)
                per_video_true_F1 = np.sum( (now_user_CU.flatten()*now_user_CU.flatten()/ now_user_CU_col_sum)/
                                            now_user_CU.flatten() + (now_user_CU.flatten()/now_user_CU_col_sum),axis = 0)/len(now_user_CU)
                True_CU[num2][:] = True_CU[num2][:] + per_video_CU/len(now_OS2)
                # TODO: Fix this
                # True_RP[num2][:] = True_RP[num2][:] +
                True_F1[num2] = True_F1[num2] + per_video_true_F1/len(now_OS2)
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