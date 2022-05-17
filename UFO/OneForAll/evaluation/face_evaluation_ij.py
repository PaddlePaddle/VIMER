"""
@author:  
@contact: 
"""
import copy
import logging
import time
import itertools
from collections import OrderedDict

import numpy as np
import paddle
import paddle.nn.functional as F
from sklearn import metrics
import sklearn
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve

from utils import comm
from utils.compute_dist import build_dist
from evaluation.evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)

def image2template_feature(img_feats=None, templates=None, medias=None):
    """image2template_feature
    """
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    print(unique_templates.shape, template_feats.shape)
    for count_template, uqt in enumerate(unique_templates):
        (ind_t, ) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m, ) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    """
    Compute set-to-set Similarity Score.
    """
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1), ))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def evaluate(embeddings, dict_label):
    """evaluate
    """
    img_feats = embeddings.cpu().numpy().astype(np.float32)
    templates = dict_label['templates']
    medias = dict_label['medias']
    p1 = dict_label['p1']
    p2 = dict_label['p2']
    label = dict_label['label']
    # get template features
    template_norm_feats, unique_templates = image2template_feature(img_feats, templates, medias)
    # get templates similarity scores
    score = verification(template_norm_feats, unique_templates, p1, p2)
    # get roc curves and tpr@fpr
    fpr, tpr, thr = roc_curve(label, score)
    return fpr, tpr, thr


class IJBCEvaluatorSingleTask(DatasetEvaluator):
    """
    FaceEvaluatorSingleTask
    """
    def __init__(self, cfg, labels, num_valid_samples, output_dir=None, **kwargs):
        """
        init
        """
        self.cfg = cfg
        self._labels = labels
        self._output_dir = output_dir
        self._cpu_device = paddle.CPUPlace()
        self._predictions = []
        self._num_valid_samples = num_valid_samples

    def reset(self):
        """
        reset
        """
        self._predictions = []

    def process(self, inputs, outputs):
        """
        process
        """
        # remove task dict
        assert len(inputs) == 1, 'support only single task evaluation'
        assert len(outputs) == 1, 'support only single task evaluation'

        inputs = list(inputs.values())[0]
        outputs = list(outputs.values())[0]

        prediction = {
            'feats': paddle.to_tensor(outputs, place=self._cpu_device, dtype=paddle.float32),
            'scores': paddle.to_tensor(inputs['targets'], place=self._cpu_device, dtype=paddle.float32),
        }
        self._predictions.append(prediction)

    def evaluate(self):
        """
        evaluate function
        """
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        features = []
        scores = []
        for prediction in predictions:
            features.append(prediction['feats'])
            scores.append(prediction['scores'])

        features = paddle.concat(features, axis=0)
        scores = paddle.concat(scores, axis=0)
        features = features[:self._num_valid_samples]
        scores = scores[:self._num_valid_samples]
        features = F.normalize(features, p=2, axis=1)
        features = features * scores.reshape((-1, 1))         
        
        self._results = OrderedDict()
        fpr, tpr, thr = evaluate(features, self._labels)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)
        thr = np.flipud(thr)
        x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2]
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
            key = 'TPR@FPR' + str(x_labels[fpr_iter])
            self._results[key] = tpr[min_index]

        return copy.deepcopy(self._results)
