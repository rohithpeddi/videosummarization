import numpy as np
import os
import json
import chainer
import skvideo
import skvideo.io as skio

from chainer import serializers
from chainer import configuration
from feature_extractor.Chain import vid_enc, vid_enc_vgg19
from feature_extractor.Chain import sampling
from feature_extractor.Chain import summarize


# settings
feat_type = 'smt_feat' # smt_feat (proposed) or vgg

# load embedding model
if feat_type == 'smt_feat':
    model = vid_enc.Model()
    serializers.load_npz('data/trained_model/model_par', model)
elif feat_type == 'vgg':
    model = vid_enc_vgg19.Model()
else:
    raise RuntimeError('[invalid feat_type] use smt_feat or vgg')


# prepair output dir
d_name = 'summe'
dataset_root = 'data/{}/'.format(d_name)
out_dir = 'results/{:}/{:}/'.format(d_name, feat_type)
print ('save to: ', out_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# load dataset metadata
dataset = json.load(open(dataset_root + 'dataset.json'))
video_id = [d['videoID'] for d in dataset]

print ('Video list:')
for vi in video_id:
    print ('-', vi)


# summarize video
v_id = 'Cooking'

with configuration.using_config('train', False):
    with chainer.no_backprop_mode():
        vsum = sampling.VSUM(v_id, model, ds=d_name, seg_l=5)

_, frames, _ = vsum.summarizeRep(seg_l=5, weights=[1.0, 0.0])

# get 0/1 label for each frame
fps = vsum.dataset.data['fps']
fnum = vsum.dataset.data['fnum']
label = summarize.get_flabel(frames, fnum, fps, seg_l=5)


# write summarized video
#
# skvideo.setFFmpegPath("D:/UTD/SEM4/RIYER/Project/ffmpeg/bin")
#
# print(skvideo.getFFmpegPath())


video_path = 'summe/videos/'+ v_id + '.webm'

video_data = skio.vread(video_path)
sum_vid = video_data[label.ravel().astype(np.bool), :,:,:]
(d1, d2, d3, d4) = sum_vid.shape

print('writing video to', out_dir + 'sum_'+ v_id + '.mp4')
# skio.vwrite('D:/UTD/SEM4/RIYER/Project/videosummarization/' + 'sum_'+ v_id + '.mp4', sum_vid)

writer = skvideo.io.FFmpegWriter(out_dir + 'sum_'+ v_id + '.mp4')
for i in range(d1):
    # print("Writing frame : " + str(i))
    writer.writeFrame(sum_vid[i, :, :, :])
writer.close()