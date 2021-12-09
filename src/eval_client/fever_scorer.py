# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/8/24
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/9/1
"""

import os
import json
import numpy as np
from collections import defaultdict
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
try:
    from .scorer import fever_score
except:
    from scorer import fever_score


prefix = os.environ['PJ_HOME']


class FeverScorer:
    def __init__(self):
        self.id2label = {2: 'SUPPORTS', 0: 'REFUTES', 1: 'NOT ENOUGH INFO'}
        self.label2id = {value: key for key, value in self.id2label.items()}

    def get_scores(self, predicted_file, actual_file=f'{prefix}/data/fever/shared_task_dev.jsonl'):
        id2results = defaultdict(dict)

        with tf.io.gfile.GFile(predicted_file) as f:
            for line in f:
                js = json.loads(line)
                guid = js['id']
                id2results[guid] = js

        with tf.io.gfile.GFile(actual_file) as fin:
            for line in fin:
                line = json.loads(line)
                guid = line['id']
                evidence = line['evidence']
                label = line['label']
                id2results[guid]['evidence'] = evidence
                id2results[guid]['label'] = label

        results = self.label_score(list(id2results.values()))
        score, accuracy, precision, recall, f1 = fever_score(list(id2results.values()))
        results.update({
            'Evidence Precision': precision,
            'Evidence Recall': recall,
            'Evidence F1': f1,
            'FEVER Score': score,
            'Label Accuracy': accuracy
        })

        return results

    def label_score(self, results):
        truth = np.array([v['label'] for v in results])
        prediction = np.array([v['predicted_label'] for v in results])
        labels = list(self.label2id.keys())
        results = {}
        p, r, f, _ = precision_recall_fscore_support(truth, prediction, labels=labels)
        for i, label in enumerate(self.label2id.keys()):
            results['{} Precision'.format(label)] = p[i]
            results['{} Recall'.format(label)] = r[i]
            results['{} F1'.format(label)] = f[i]

        return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_file", '-i', type=str)
    args = parser.parse_args()

    scorer = FeverScorer()
    results = scorer.get_scores(args.predicted_file)
    print(json.dumps(results, ensure_ascii=False, indent=4))
