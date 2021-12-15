# -*- coding: utf-8 -*-

"""
@Author     : Jiangjie Chen
@Time       : 2020/12/31 18:38
@Contact    : jjchen19@fudan.edu.cn
@Description:
"""

import tensorflow as tf
import cjjpy as cjj
import ujson as json
from hparams import *
import sys, os
sys.path.append('..')
from dataloaders import FEVERLoader


def rewrite_file(filename, loader):
    with tf.io.gfile.GFile(filename) as f:
        data = f.readlines()

    with tf.io.gfile.GFile(filename, 'w') as fo:
        for line in data:
            js = json.loads(line)
            if js.get('predicted_evidence') is None:
                js['predicted_evidence'] = [[ev[0], ev[1]] for ev in loader[js['id']]['bert_evidence']]
            fo.write(json.dumps(js) + '\n')
        print(f'* {filename} rewritten')


for role in ['eval', 'test']:
    floader = FEVERLoader(role)
    floader.load_fever('bert', clean_load=False)
    filename = os.path.join(AG_PREFIX.format(version='v5'), CACHED_EVIDENTIAL_FILE.format(role=role, k_cand=4))
    rewrite_file(filename, floader)
    final_output = os.path.join(cjj.AbsParentDir(AG_PREFIX.format(version='v5'), '.'),
                                FINAL_FILE.format(role=role))
    rewrite_file(final_output, floader)