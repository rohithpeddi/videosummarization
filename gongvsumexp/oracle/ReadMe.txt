#### General ####  
1. We work on "OVP" and "YouTube" datasets for video summarization

2. The original videos, user ground-truths, and evaluation package can be downloaded from
https://sites.google.com/site/vsummsite/

3. Here we provide the "ground set" we downsampled from the videos and the single "oracle" we generate for each video for training purpose.
Note that in evaluation, we compare the summarization results to all the user summaries following the VSUMM paper.


#### Ground set & oracle ####  
4. In /Oracle_groundset, each mat file is a 3x50 cell.
The 1st column corresponds to the video index (Note that OVP with index from 21~70; YouTube 11~20 and 71~110)
The 2nd column is the oracle frame indices of the original video
The 3rd column is the ground set frame indices of the original video
Note that the 2nd column is always a subset of the 3rd column
** In our papers, we examine on all 50 OVP videos and 39 YouTube videos (71~81 and 83~110). **


#### Features ####  
5. In /feature, we provide for each video several type of features.
Note that the features are extracted only for those "ground set" frames.
To get the features for "oracle" frames, you can do the following:
If the oracle is {15, 75} and the ground set is {15, 30, 45, 60, 75, 90}, then the 1st and 5th feature vectors correspond to the oracle frames.

6. In our large-margin DPP paper (http://auai.org/uai2015/proceedings/papers/209.pdf), we use "saliency" and "context2" for quality features.
We then use "hist_5_5_16", "hist_1_1_16", and "fishers" for similarity features.

7. In our sequential DPP paper (http://papers.nips.cc/paper/5413-large-scale-l-bfgs-using-mapreduce), we concatenate "fishers_PCA90", "saliency", and "context2" to be the features.


#### Evaluation ####  
8. The VSUMM evaluation metric is content-based.
That is, you don't need to summarize your video by exact the same frame indices.
You need to follow the protocol to generate a folder and put your summarized frames in.
 