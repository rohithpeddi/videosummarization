import json
import numpy as np

datasetRoot = 'data/summe/'

class SUMMDATA():

    def __init__(self, videoId, featType='vgg'):
        dataset = json.load(open(datasetRoot + 'dataset.json'))
        print('load ' + videoId)
        data = list(filter(lambda x: x['videoID'] == videoId, dataset))
        self.data = data[0]
        self.feat = np.load(datasetRoot + 'feat/' + featType + '/' + videoId + '.npy').astype(np.float32)

    def sampleFrame(self):
        fps = self.data['fps']
        fnum = self.data['fnum']

        idx = np.arange(fps, fnum, fps)
        idx = np.floor(idx)
        idx = idx.tolist()
        # idx = map(int, idx)

        img = []
        img_id = []
        score = []
        for i in range(len(idx)):
            index = int(idx[i])
            img.append(self.data['image'][index])
            img_id.append(self.data['imgID'][index])
            score.append(self.data['score'][index])

        return img, img_id, score