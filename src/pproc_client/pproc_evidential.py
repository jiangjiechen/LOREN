# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/8/16 16:51
@Contact    : jjchen19@fudan.edu.cn
@Description: 
'''

import cjjpy as cjj
import sys
import tensorflow as tf
import ujson as json
import argparse

try:
    from .hparams import *
    from ..dataloaders import FEVERLoader
    from ..mrc_client.answer_generator import AnswerGenerator, assemble_answers_to_one
except:
    sys.path.append(cjj.AbsParentDir(__file__, '..'))
    from hparams import *
    from dataloaders import FEVERLoader
    from mrc_client.answer_generator import AnswerGenerator, assemble_answers_to_one


def prepare_evidential(version, role, mrc_model_path, evi_key,
                       k_cand=4, batch_size=64):
    '''
    After pproc_questions (prepare_answers, prepare_questions)
    :return:
    {
        'id': id,
        'label': x,
        'claim': c,
        'evidence': [e1, e2, ...],
        'answers': [a1, a2, ...],
        'questions': [q1, q2, ...],
        'cloze_qs': [q1, q2, ...], #m
        'regular_qs': [q1, q2, ...], #m
        'answer_roles': [noun, noun, adj, verb, ...], # m
        'evidential': [[b1, b2, ... bk]_1, [...]_2, ...],
        'evidential_assembled': [], # m
    }
    '''
    tf.io.gfile.makedirs(AG_PREFIX.format(version=version))
    cached_evidential = AG_PREFIX.format(version=version) \
                        + CACHED_EVIDENTIAL_FILE.format(k_cand=k_cand, role=role)
    cached_question = QG_PREFIX.format(version=version) + CACHED_QUESTION_FILE.format(role=role)

    ag = AnswerGenerator(mrc_model_path)
    ag.init_model()
    with tf.io.gfile.GFile(cached_question) as f, \
            tf.io.gfile.GFile(cached_evidential, 'w') as fo:
        data = f.read().splitlines()
        examples, ids = [], []
        data_dict = {}
        for line in data:
            js = json.loads(line)
            data_dict[js['id']] = js
            for q in js['questions']:
                ids.append(js['id'])
                ex = ag.assemble(q, " ".join(js["evidence"]))
                examples.append(ex)

        predicted = ag.generate(examples, num_beams=k_cand, num_return_sequences=k_cand,
                                 batch_size=batch_size, verbose=True)
        assert len(predicted) == len(examples)

        # follow by strict order
        for answers, id in zip(predicted, ids):
            if 'evidential' in data_dict[id]:
                # [b1, b2, ..., bk]
                data_dict[id]['evidential'].append(answers)
            else:
                data_dict[id]['evidential'] = [answers]

        _ = [_sanity_check(data_dict[k]) for k in data_dict]

        if role in ['eval', 'test']:
            floader = FEVERLoader(role)
            print('Loading FEVER...')
            floader.load_fever(evi_key.split('_')[0], clean_load=False)

        for k in data_dict:
            js = data_dict[k]
            if role in ['eval', 'test']:
                if js.get('predicted_evidence') is None:
                    js['predicted_evidence'] = [[ev[0], ev[1]] for ev in floader[js['id']][evi_key]]
            fo.write(json.dumps(js) + '\n')

    final_output = os.path.join(cjj.AbsParentDir(AG_PREFIX.format(version=version), '.'),
                                FINAL_FILE.format(role=role))

    tf.io.gfile.copy(cached_evidential, final_output)

    cjj.lark(f'Final baked in {final_output}')
    return final_output


def _sanity_check(js):
    assert len(js['evidential']) == len(js['questions']) == len(js['answers']), js


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_cand', '-k', type=int, default=4,
                        help='number of candidate answer')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--version', '-v', type=str, help='v1, v2, ...', default='v5')
    parser.add_argument('--roles', nargs='+', required=True,
                        help='train val test eval')
    parser.add_argument('--evi_key', '-e', type=str, choices=['bert_evidence'], default='bert_evidence')
    parser.add_argument('--mrc_model_name', '-m', type=str, required=True,
                        help='Absolute path of the mrc model')
    args = parser.parse_args()

    server = None

    for role in args.roles:
        evidential_output = prepare_evidential(args.version, role, args.mrc_model_name, args.evi_key,
                                               args.k_cand, args.batch_size)
