# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/9/7
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/9/7
"""

import json
import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

# ref --> label 1, nei & sup --> label 0
idx2label = {0: 1, 1: 0, 2: 0}


def read_json_lines(filename, mode='r', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fin:
        for line in fin:
            if skip > 0:
                skip -= 1
                continue
            yield json.loads(line)


def process(filein):
    id2info = defaultdict(dict)
    for line in read_json_lines('eval.human.ref.merged.json'):
        labels = [0] * len(line['questions'])
        for cul in line['culprit']:
            labels[cul] = 1
        id2info[line['id']].update({'id': line['id'], 'labels': labels})

    for line in read_json_lines(filein):
        if line['id'] not in id2info: continue
        predicted = [idx2label[_] for _ in np.argmax(line['z_prob'], axis=-1)]
        id2info[line['id']]['predicted'] = predicted

    ps, rs, fs = [], [], []
    for info in id2info.values():
        p, r, f, _ = precision_recall_fscore_support(info['labels'], info['predicted'], average='binary')
        ps.append(p)
        rs.append(r)
        fs.append(f)
    print(filein)
    print('Precision: {}'.format(sum(ps) / len(ps)))
    print('Recall: {}'.format(sum(rs) / len(rs)))
    print('F1: {}'.format(sum(fs) / len(fs)))

    return sum(ps) / len(ps), sum(rs) / len(rs), sum(fs) / len(fs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='predicted jsonl file with phrasal veracity predictions.')
    args = parser.parse_args()
    process(args.i)
