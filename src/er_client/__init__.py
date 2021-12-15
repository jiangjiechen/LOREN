# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2020/9/21 16:13
@Contact    : jjchen19@fudan.edu.cn
@Description:
'''

import cjjpy as cjj
import os
# from .document_retrieval import DocRetrieval
from .doc_retrieval_by_api import DocRetrieval
from .sentence_selection import SentSelector


arg_values = {
    'batch_size': 32,
    'dropout': 0.6,
    'use_cuda': True,
    'bert_hidden_dim': 768,
    'layer': 1,
    'num_labels': 3,
    'evi_num': 5,
    'threshold': 0.0,
    'max_len': 120,
}

args = cjj.AttrDict(arg_values)

class EvidenceRetrieval:
    def __init__(self, er_model_dir=cjj.AbsParentDir(__file__, '...', 'models/evidence_retrieval/')):
        # self.doc_retriever = DocRetrieval(cjj.AbsParentDir(__file__, '...', 'data/fever.db'),
        #                                   add_claim=True, k_wiki_results=7)
        self.doc_retrieval = DocRetrieval(link_type='tagme')
        self.sent_selector = SentSelector(os.path.join(er_model_dir, 'bert_base/'),
                                          os.path.join(er_model_dir, 'retrieval_model/model.best.pt'),
                                          args)

    def retrieve(self, claim):
        # noun_phrases, wiki_results, predicted_pages = self.doc_retriever.exact_match(claim)
        # evidence = []
        # for page in predicted_pages:
        #     evidence.extend(self.doc_retriever.db.get_doc_lines(page))
        evidence = self.doc_retrieval.retrieve_docs(claim)
        evidence = self.rank_sentences(claim, evidence)
        return evidence

    def rank_sentences(self, claim, sentences, id=None):
        '''
        :param claim: str
        :param sentences: [(ent, num, sent) * N]
        :param id:
        :return: [(ent, num, sent) * k]
        '''
        if id is None:
            id = len(claim)

        result = self.sent_selector.rank_sentences([{'claim': claim,
                                                     'evidence': sentences,
                                                     'id': id}])
        evidence = result.get(id, [])
        return evidence