# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/7/25 18:23
@Contact    : jjchen19@fudan.edu.cn
@Description:
'''

import os
import cjjpy as cjj
import sys
import tensorflow as tf
import ujson as json
from tqdm import tqdm
import argparse

try:
    sys.path.append(cjj.AbsParentDir(__file__, '..'))
    from hparams import *
    from pseudo_multiproc_toolkit import *
    from dataloaders import FEVERLoader
    from parsing_client.sentence_parser import SentenceParser, deal_bracket
    from qg_client.question_generator import QuestionGenerator
except:
    from .hparams import *
    from .pseudo_multiproc_toolkit import *
    from ..dataloaders import FEVERLoader
    from ..parsing_client.sentence_parser import SentenceParser, deal_bracket
    from ..qg_client.question_generator import QuestionGenerator


def prepare_answers(version, role, evi_key='bert_evidence', overwrite=False):
    '''
    :return
    {
        'id': id,
        'claim': c,
        'label': x,
        'evidence': [e1, e2, ...], # n
        'answers': [a1, a2, ...], # m
        'answer_roles': [noun, noun, adj, verb, ...] # m
    }
    '''
    assert role in ['val', 'test', 'train', 'eval'], role

    def _proc_one(js):
        js.pop('all_evidence')
        evidence = [deal_bracket(ev[2], True, ev[0]) for ev in js[evi_key]]
        results = sent_client.identify_NPs(deal_bracket(js['claim'], True),
                                           candidate_NPs=[x[0] for x in js[evi_key]])
        NPs = results['NPs']
        claim = results['text']
        verbs = results['verbs']
        adjs = results['adjs']
        _cache = {'id': js['id'],
                  'claim': claim,
                  'evidence': evidence,
                  'answers': NPs + verbs + adjs,
                  'answer_roles': ['noun'] * len(NPs) + ['verb'] * len(verbs) + ['adj'] * len(adjs)}
        if js.get('label'):
            _cache.update({'label': js['label']})
        return _cache

    cached_ = QG_PREFIX.format(version=version) + CACHED_ANSEWR_FILE.format(role=role)
    tf.io.gfile.makedirs(QG_PREFIX.format(version=version))
    if tf.io.gfile.exists(cached_) and not overwrite:
        print(f'* Skipped, exising {cached_}')
        return cached_

    sent_client = SentenceParser(device='cuda:0')
    floader = FEVERLoader(role)
    floader.load_fever(evi_key.split('_')[0])

    with tf.io.gfile.GFile(cached_, 'w') as f:
        for id in tqdm(floader, desc=f'{role} answer'):
            res = _proc_one(floader[id])
            f.write(json.dumps(res) + '\n')

    cjj.lark(f'* NPs baked in {cached_}')
    return cached_


def prepare_questions(version, role, qg_model='t5', batch_size=64, overwrite=False):
    '''
    After prepare_nps
    :return
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
    cached_answer = QG_PREFIX.format(version=version) + CACHED_ANSEWR_FILE.format(role=role)
    cached_question = QG_PREFIX.format(version=version) + CACHED_QUESTION_FILE.format(role=role)

    if tf.io.gfile.exists(cached_question) and not overwrite:
        print(f'* Skipped, existing {cached_question}')
        return cached_question

    qg_client = QuestionGenerator(qg_model)
    with tf.io.gfile.GFile(cached_answer, 'r') as f, \
            tf.io.gfile.GFile(cached_question, 'w') as fo:
        data = f.read().splitlines()
        data_dict = {}
        _cache = []
        for line in data:
            js = json.loads(line)
            data_dict[js['id']] = js
            if len(js['answers']) == 0:
                # TODO: hack empty answer
                print('Empty answer:', js)
                pseudo_answer = js['claim'].split()[0]
                js['answers'] = [(pseudo_answer, 0, len(pseudo_answer))]
                js['answer_roles'] = ['noun']
            for answer in js['answers']:
                _cache.append((js['claim'], [answer], js['id']))
        print(_cache[:5])

        qa_pairs = qg_client.generate([(x, y) for x, y, z in _cache], batch_size=batch_size)
        print(qa_pairs[:5])

        for (q, clz_q, a), (_, _, id) in zip(qa_pairs, _cache):
            if 'questions' in data_dict[id]:
                data_dict[id]['cloze_qs'].append(clz_q)
                data_dict[id]['regular_qs'].append(q)
                data_dict[id]['questions'].append(qg_client.assemble_question(q, clz_q))
            else:
                data_dict[id]['cloze_qs'] = [clz_q]
                data_dict[id]['regular_qs'] = [q]
                data_dict[id]['questions'] = [qg_client.assemble_question(q, clz_q)]

        _ = [_sanity_check(data_dict[k]) for k in data_dict]

        for k in data_dict:
            fo.write(json.dumps(data_dict[k]) + '\n')

    cjj.lark(f'* Questions baked in {cached_question}')
    return cached_question


def _sanity_check(js):
    try:
        assert len(js['questions']) == len(js['answers'])
        assert len(js['answers']) == len(js['answer_roles'])
    except:
        print(js)
        raise Exception


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--evi_key', '-e', type=str, default='bert_evidence')
    parser.add_argument('--version', '-v', type=str, help='v1, v2, ...', default='v5')
    parser.add_argument('--roles', nargs='+', required=True,
                        help='train val test eval')
    parser.add_argument('--qg_model', '-m', type=str, default='t5')
    args = parser.parse_args()

    for role in args.roles:
        prepare_answers(args.version, role, args.evi_key, args.overwrite)
        prepare_questions(args.version, role, args.qg_model, args.batch_size, args.overwrite)
