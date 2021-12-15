# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/9/20 11:42
@Contact    : jjchen19@fudan.edu.cn
@Description: 
'''

import torch
from transformers import BertTokenizer
from .retrieval_model.bert_model import BertForSequenceEncoder
from .retrieval_model.models import inference_model
from .retrieval_model.data_loader import DataLoaderTest


class SentSelector:
    def __init__(self, pretrained_bert_path, select_model_path, args):
        self.args = args
        self.use_cuda = self.args.use_cuda and torch.cuda.is_available()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.bert_model = BertForSequenceEncoder.from_pretrained(pretrained_bert_path)

        self.rank_model = inference_model(self.bert_model, self.args)
        self.rank_model.load_state_dict(torch.load(select_model_path,
                                                   map_location=None if self.use_cuda else torch.device('cpu'))['model'])

        if self.use_cuda:
            self.bert_model = self.bert_model.cuda()
            self.rank_model.cuda()

    def rank_sentences(self, js: list):
        '''
        :param js: [{'claim': xxx, 'id': xx, 'evidence': xxx}]
        :return: [(ent, num, sent, prob), (ent, num, sent, prob)]
        '''
        data_reader = DataLoaderTest(js, self.tokenizer, self.args, self.use_cuda)
        self.rank_model.eval()
        all_predict = dict()
        for inp_tensor, msk_tensor, seg_tensor, ids, evi_list in data_reader:
            probs = self.rank_model(inp_tensor, msk_tensor, seg_tensor)
            probs = probs.tolist()
            assert len(probs) == len(evi_list)
            for i in range(len(probs)):
                if ids[i] not in all_predict:
                    all_predict[ids[i]] = []
                # if probs[i][1] >= probs[i][0]:
                all_predict[ids[i]].append(tuple(evi_list[i]) + (probs[i],))

        results = {}
        for k, v in all_predict.items():
            sorted_v = sorted(v, key=lambda x: x[-1], reverse=True)
            results[k] = sorted_v[:self.args.evi_num]
        return results
