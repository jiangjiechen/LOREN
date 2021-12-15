# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/7/20 17:54
@Contact    : jjchen19@fudan.edu.cn
@Description:
'''

import os
import ujson as json
import tensorflow as tf
import argparse
import random
from transformers import BartTokenizer

try:
    from .hparams import CACHED_QUESTION_FILE, QG_PREFIX
except:
    from hparams import CACHED_QUESTION_FILE, QG_PREFIX


random.seed(1111)


def pproc_seq2seq(input_file, output_dir, role):
    '''
    :param input_file:
        {
        'id': id,
        'claim': c,
        'label': x,
        'evidence': [e1, e2, ...], # n
        'answers': [a1, a2, ...], # m
        'questions': [q1, q2, ...], # m
        'cloze_qs': [q1, q2, ...], #m
        'regular_qs': [q1, q2, ...], #m
        'answer_roles': [noun, noun, adj, verb, ...] # m
    }
    '''
    assert role in ['val', 'test', 'train'], role

    use_rag = 'v6' in input_file
    if not use_rag:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    tf.io.gfile.makedirs(output_dir)
    src_fname = os.path.join(output_dir, f'{role}.source')
    tgt_fname = os.path.join(output_dir, f'{role}.target')

    with tf.io.gfile.GFile(input_file) as fin, \
            tf.io.gfile.GFile(src_fname, 'w') as srcf, \
            tf.io.gfile.GFile(tgt_fname, 'w') as tgtf:
        data = fin.readlines()
        for line in data:
            js = json.loads(line)
            if js['label'] == 'SUPPORTS':
                evidence = ' '.join(js['evidence'])
                questions = js['questions']
                i = random.randint(0, len(questions) - 1)
                if use_rag:
                    srcf.write(f'{questions[i]}\n')
                else:
                    srcf.write(f'{questions[i]} {tokenizer.sep_token} {evidence}\n')
                tgtf.write(js['answers'][i][0] + '\n')

    return src_fname, tgt_fname


def pproc_for_mrc(output_dir, version):
    assert version in ['v5']
    for role in ['val', 'train', 'test']:
        _role = 'val' if role == 'test' else role
        input_file = os.path.join(QG_PREFIX.format(version=version),
                                  CACHED_QUESTION_FILE.format(role=_role))
        pproc_seq2seq(input_file, output_dir, role)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-o', required=True, default='data/mrc_seq2seq_v5',
                        help='data path, e.g. data/mrc_seq2seq_v5')
    parser.add_argument('--version', '-v', type=str, default='v5')
    args = parser.parse_args()
    pproc_for_mrc(args.output_dir, args.version)
