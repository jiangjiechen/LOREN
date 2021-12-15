# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/7/20 17:34
@Contact    : jjchen19@fudan.edu.cn
@Description: 
'''

import tensorflow as tf
import cjjpy as cjj
import os
import re
import ujson as json
from collections import defaultdict

pj_prefix = cjj.AbsParentDir(__file__, '..')


class FEVERLoader:
    def __init__(self, role):
        role = 'dev' if role == 'val' else role
        assert role in ['train', 'dev', 'test', 'eval']
        self.role = role
        self.fever_data = defaultdict(dict)
        self.SUPPORTS = 'SUPPORTS'
        self.REFUTES = 'REFUTES'
        self.NEI = 'NOT ENOUGH INFO'

    def __iter__(self):
        for k in self.fever_data:
            yield k

    def __len__(self):
        return len(self.fever_data)

    def __getitem__(self, item):
        return self.fever_data[item]

    def load_fever(self, retrieve_type='bert', clean_load=True):
        self._load_fever_golden()
        self._load_fever_all()
        self._load_fever_retrieved(retrieve_type, clean_load)

    def _load_json(self, fname):
        with tf.io.gfile.GFile(fname) as f:
            return [json.loads(x) for x in f.readlines()]

    def _new_role(self):
        role = self.role if self.role != 'eval' else 'dev'
        return role

    def _load_fever_golden(self):
        if self.role == 'test':
            postfix = f'data/fever/shared_task_test.jsonl'
            for js in self._load_json(f'{pj_prefix}/{postfix}'):
                self.fever_data[js['id']].update({
                    'id': js['id'],
                    'claim': js['claim']
                })
        else:
            role = self._new_role()
            postfix = f'data/fever/baked_data/golden_{role}.json'
            for js in self._load_json(f'{pj_prefix}/{postfix}'):
                self.fever_data[js['id']].update({
                    'id': js['id'],
                    'claim': js['claim'],
                    'label': js['label'],
                    'golden_evidence': self._clean_evidence(js['evidence'])
                })
        print('* FEVER golden loaded.')

    def _load_fever_all(self):
        role = self._new_role()
        postfix = f'data/fever/baked_data/all_{role}.json'
        for js in self._load_json(f'{pj_prefix}/{postfix}'):
            self.fever_data[js['id']].update({
                'all_evidence': self._clean_evidence(js['evidence'])
            })
        print('* FEVER all loaded.')

    def _load_fever_retrieved(self, retrieve_type, clean_load):
        assert retrieve_type in ['bert']
        postfix = f'data/fever/baked_data/{retrieve_type}_{self.role}.json'
        for js in self._load_json(f'{pj_prefix}/{postfix}'):
            self.fever_data[js['id']].update({
                f'{retrieve_type}_evidence': self._clean_evidence(js['evidence']) if clean_load else js['evidence']
            })
        print(f'* FEVER {retrieve_type} loaded.')

    def clean_text(self, sentence):
        sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
        sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
        sentence = re.sub(" -LRB-", " ( ", sentence)
        sentence = re.sub("-RRB-", " )", sentence)

        sentence = re.sub(" LSB.*?RSB", "", sentence)
        sentence = re.sub("LRB RRB ", "", sentence)
        sentence = re.sub("LRB", " ( ", sentence)
        sentence = re.sub("RRB", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)
        sentence = re.sub('  ', ' ', sentence)
        return sentence

    def clean_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub(" -LRB-", " ( ", title)
        title = re.sub("-RRB-", " )", title)
        title = re.sub("-COLON-", ":", title)
        title = re.sub('  ', ' ', title)
        return title

    def _clean_evidence(self, evidence):
        cev = []
        for ev in evidence:
            if len(ev) == 4:
                cev.append([self.clean_title(ev[0]), ev[1], self.clean_text(ev[2]), ev[3]])
            elif len(ev) == 3:
                cev.append([self.clean_title(ev[0]), ev[1], self.clean_text(ev[2])])
            elif len(ev) == 0:
                cev.append(ev)
            else:
                raise ValueError(ev)
        return cev


if __name__ == '__main__':
    floader = FEVERLoader('test')
    floader.load_fever('bert', clean_load=False)
    for k in floader:
        print(floader[k])
        input()
